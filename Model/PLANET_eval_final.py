import torch
import numpy as np
import argparse
import os
import sys
import json
import gzip
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import math
import wandb
from rdkit import Chem, RDLogger
from collections import defaultdict
import pickle
import random
import re
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- PATH AYARLARI ---
MODEL_DIR_PATH = os.path.join(PLANET_ROOT, "Model") 
PCA_PARAMS_PATH = os.path.join(PLANET_ROOT, "esm2_pca_params.pkl")

if MODEL_DIR_PATH not in sys.path: sys.path.append(MODEL_DIR_PATH)

try:
    from PLANET_model_esm2 import PLANET
    print("✅ PLANET_model_esm2 YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model_esm2.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# 🧬 YARDIMCI FONKSİYONLAR (MOL2 & PDB DESTEKLİ)
# ==============================================================================
AA_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
LIGADND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1 
BOND_FDIM = 6  

def get_coords_from_line(line):
    """PDB veya MOL2 satırından koordinatları akıllıca çeker."""
    try:
        if line.startswith("ATOM") or line.startswith("HETATM"): # PDB
            return [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        else: # MOL2 (Sütun bazlı)
            parts = line.split()
            if len(parts) >= 6: return [float(parts[2]), float(parts[3]), float(parts[4])]
    except: return None
    return None

def mass_center_generic(lines):
    """Her iki format için de çalışan ağırlık merkezi ve CA tespiti."""
    coords, alphas = [], []
    for line in lines:
        c = get_coords_from_line(line)
        if c:
            coords.append(c)
            # CA kontrolü PDB için spesifik sütuna, MOL2 için isme bakar
            if " CA " in line or (not line.startswith("ATOM") and "CA" in line):
                alphas.append(c)
    coords = np.array(coords, dtype=np.float32)
    center = np.mean(coords, axis=0, keepdims=True) if len(coords) > 0 else np.zeros((1,3))
    alpha_pos = np.array(alphas, dtype=np.float32) if len(alphas) > 0 else center
    return center, coords, alpha_pos

def parse_mapping_generic(content_lines):
    """Protein dizisi ve ESM index eşleşmesini her iki formatta çıkarır."""
    mapping, idx, seen = {}, 0, set()
    for line in content_lines:
        try:
            if line.startswith("ATOM"): # PDB
                res_num = int(re.search(r'\d+', line[22:26]).group())
                chain, res_name = line[21].strip() or 'A', line[17:20].strip()
            else: # MOL2
                parts = line.split()
                if len(parts) < 8: continue
                res_num, res_name, chain = int(parts[6]), parts[7][:3], 'A'
            key = (chain, res_num)
            if key not in seen and res_name in AA_3_TO_1:
                mapping[key] = idx
                seen.add(key); idx += 1
        except: continue
    return mapping

# --- PCA LOAD ---
PCA_MEAN, PCA_TRANSFORM = None, None
def load_pca_params():
    global PCA_MEAN, PCA_TRANSFORM
    if os.path.exists(PCA_PARAMS_PATH):
        try:
            with open(PCA_PARAMS_PATH, 'rb') as f:
                params = pickle.load(f)
                if isinstance(params, dict): PCA_MEAN = params['mean']; PCA_TRANSFORM = params['components']
                else: PCA_MEAN = params.mean_; PCA_TRANSFORM = params.components_
            print("✅ PCA Parametreleri yüklendi.")
        except: pass

def apply_pca_transform_single(embedding_1280d):
    if PCA_TRANSFORM is None: return embedding_1280d[:512]
    return np.dot(embedding_1280d - PCA_MEAN, PCA_TRANSFORM.T).astype(np.float32)

# ==============================================================================
# 🛡️ GÜNCELLENMİŞ SINIFLAR
# ==============================================================================

class Residue():
    def __init__(self, residue_content): 
        self.mass_center, self.coordinates, self.alpha_position = mass_center_generic(residue_content)
        self.feature = np.zeros(512, dtype=np.float32)

class PocketProcessor():
    def __init__(self, protein_path, crystal_ligand_path, esm_embedding):
        self.fresidues, self.alpha_coordinates, self.valid = [], [], False
        try:
            with open(protein_path, 'r') as f:
                all_lines = [l.strip() for l in f if len(l.strip()) > 10]
        except: return

        ligand = None
        try:
            if crystal_ligand_path.endswith('.mol2'): ligand = Chem.MolFromMol2File(crystal_ligand_path, sanitize=False)
            else: ligand = Chem.SDMolSupplier(crystal_ligand_path, sanitize=False)[0]
        except: pass
        if ligand is None: return
        
        conf = ligand.GetConformer()
        center = np.mean([conf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())], axis=0)
        
        all_coords = [get_coords_from_line(l) for l in all_lines if get_coords_from_line(l)]
        if not all_coords: return
        coords_np = np.array(all_coords)
        dists = np.sum((coords_np - center)**2, axis=1) # Broadcast hatası burada çözüldü
        
        pocket_indices = np.where(dists <= 144.0)[0]
        if len(pocket_indices) < 10: pocket_indices = np.argsort(dists)[:600]
        pocket_lines = [all_lines[i] for i in pocket_indices]

        mapping = parse_mapping_generic(all_lines)
        res_dict = defaultdict(list)
        for line in pocket_lines:
            try:
                if line.startswith("ATOM"): num, ch = int(re.search(r'\d+', line[22:26]).group()), line[21].strip() or 'A'
                else: parts = line.split(); num, ch = int(parts[6]), 'A'
                res_dict[(num, ch)].append(line)
            except: continue

        for (num, ch), r_lines in res_dict.items():
            res = Residue(r_lines)
            if esm_embedding is not None and (ch, num) in mapping:
                idx = mapping[(ch, num)]
                if idx < len(esm_embedding): res.feature = apply_pca_transform_single(esm_embedding[idx])
            self.fresidues.append(res.feature)
            pos = res.alpha_position[0] if res.alpha_position.ndim > 1 else res.alpha_position
            self.alpha_coordinates.append(pos.reshape(1,3))

        if self.fresidues:
            self.fresidues = torch.tensor(np.stack(self.fresidues)).float()
            self.alpha_coordinates = torch.tensor(np.concatenate(self.alpha_coordinates)).float()
            self.valid = True

class LigandDataset(Dataset):
    """🚀 GÜNCEL: Hem ham SDF hem GZ destekler."""
    def __init__(self, sdf_file, max_limit=None):
        self.mols = []
        try:
            open_func = gzip.open if sdf_file.endswith('.gz') else open
            with open_func(sdf_file, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f, sanitize=False)
                for mol in suppl: 
                    if mol: self.mols.append(mol)
        except Exception as e: print(f"⚠️ Ligand Okuma Hatası ({sdf_file}): {e}")
        if max_limit and len(self.mols) > max_limit:
            random.seed(42); self.mols = random.sample(self.mols, max_limit)
    def __len__(self): return len(self.mols)
    def __getitem__(self, idx): return self.mols[idx]

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGADND_ELEMENTS) + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3]) + onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other']) + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(), bond.GetIsConjugated()])

def mol_batch_to_graph(mol_batch):
    fatoms, fbonds = [], [torch.zeros(ATOM_FDIM + BOND_FDIM)] 
    in_bonds, all_bonds = [], [(-1,-1)] 
    lig_scope, total_atoms = [], 0
    for mol in mol_batch:
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms(): fatoms.append(atom_features(atom)); in_bonds.append([])
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx() + total_atoms, bond.GetEndAtomIdx() + total_atoms
            b = len(all_bonds); all_bonds.append((a1,a2)); fbonds.append(torch.cat([fatoms[a1], bond_features(bond)], 0)); in_bonds[a2].append(b)
            b = len(all_bonds); all_bonds.append((a2,a1)); fbonds.append(torch.cat([fatoms[a2], bond_features(bond)], 0)); in_bonds[a1].append(b)
        lig_scope.append((total_atoms, n_atoms)); total_atoms += n_atoms
    fatoms = torch.stack(fatoms, 0) if fatoms else torch.zeros((0, ATOM_FDIM + BOND_FDIM))
    fbonds = torch.stack(fbonds, 0); total_bonds = len(all_bonds)
    agraph, bgraph = torch.zeros(total_atoms, 10).long(), torch.zeros(total_bonds, 10).long()
    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]): 
            if i < 10: agraph[a,i] = b
    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if i < 10 and all_bonds[b2][0] != y: bgraph[b1,i] = b2
    return (fatoms, fbonds, agraph, bgraph, lig_scope)

# ==============================================================================
# 🎯 METRİKLER VE ENGINE
# ==============================================================================

def calculate_ef_metrics(active_scores, decoy_scores):
    """Eksiksiz EF hesaplayıcı (%0.5, %1, %2, %5)."""
    y_true = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_score = np.concatenate([active_scores, decoy_scores])
    sorted_labels = y_true[np.argsort(-y_score)]
    n_total, n_actives = len(y_score), len(active_scores)
    ef_results = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n_sample = math.ceil(n_total * pct)
        if n_sample == 0: ef_results[f'EF_{pct*100}%'] = 0.0; continue
        ef = (sorted_labels[:n_sample].sum() / n_sample) / (n_actives / n_total)
        ef_results[f'EF_{pct*100}%'] = round(ef, 3)
    return ef_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('-h5', '--h5_path', required=True)
    parser.add_argument('-j', '--json_path', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    load_pca_params()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    wandb.init(project="PLANET_Benchmark_FINAL", name=f"Run_{os.path.basename(args.model_path)}", config=args)

    # Model parametreleri varsayılan olarak set edildi (Bioinformatics setup)
    model = PLANET(512, 8, 128, 128, 3, 10, 1, str(device))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    with open(args.json_path, 'r') as f: data_map = json.load(f)
    h5_file = h5py.File(args.h5_path, 'r')
    csv_path = os.path.join(args.out_dir, 'benchmark_results.csv')
    
    results = []
    completed_ids = set(pd.read_csv(csv_path)['Protein_ID'].tolist()) if os.path.exists(csv_path) else set()

    for target_name, records in data_map.items():
        for record in records:
            protein_id = record.get('name', target_name)
            if protein_id in completed_ids: continue

            print(f"\n⚡ İşleniyor: {target_name} ({protein_id})")
            target_esm = h5_file[protein_id][:] if protein_id in h5_file else h5_file.get(target_name, [None])[0]
            prot_path = record.get('protein_file') or record.get('protein_pdb')
            ref_lig = record.get('ligand_mol2') or record.get('ligand_file')

            if target_esm is None or not prot_path: continue
            pocket_data = PocketProcessor(prot_path, ref_lig, target_esm)
            if not pocket_data.valid: continue

            p_fres, p_coords = pocket_data.fresidues.to(device), pocket_data.alpha_coordinates.to(device)

            def predict(loader):
                scores = []
                with torch.no_grad():
                    for batch in loader:
                        if not batch: continue
                        fatoms, fbonds, agraph, bgraph, lig_scope = batch
                        bs = len(lig_scope); n_res = p_fres.shape[0]
                        res_in = (p_fres.repeat(bs, 1), None, [(i*n_res, n_res) for i in range(bs)], p_coords.repeat(bs, 1))
                        mol_in = (fatoms.to(device), fbonds.to(device), agraph.to(device), bgraph.to(device), lig_scope)
                        _, _, aff = model(res_in, mol_in)
                        scores.append(aff.cpu().numpy().flatten())
                return np.concatenate(scores) if scores else []

            act_loader = DataLoader(LigandDataset(record['actives_file']), batch_size=args.batch_size, collate_fn=lambda x: mol_batch_to_graph([m for m in x if m]))
            dec_loader = DataLoader(LigandDataset(record['decoys_file']), batch_size=args.batch_size, collate_fn=lambda x: mol_batch_to_graph([m for m in x if m]))

            act_scores, dec_scores = predict(act_loader), predict(dec_loader)

            if len(act_scores) > 0 and len(dec_scores) > 0:
                efs = calculate_ef_metrics(act_scores, dec_scores)
                auc_val = roc_auc_score(np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)]), np.concatenate([act_scores, dec_scores]))
                log_row = {"Target": target_name, "Protein_ID": protein_id, "ROC_AUC": auc_val}
                log_row.update(efs)
                print(f"✅ {protein_id} -> AUC: {auc_val:.4f} | EF_1%: {efs['EF_1.0%']:.2f}")
                results.append(log_row); wandb.log(log_row)
                pd.DataFrame(results).to_csv(csv_path, index=False)

    h5_file.close(); wandb.finish()