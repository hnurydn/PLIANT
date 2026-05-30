import torch
import numpy as np
import argparse
import os
import sys
import json
import gzip
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import math
import wandb
from rdkit import Chem, RDLogger
from collections import defaultdict
import pickle
import random
import re

# --- PATH AYARLARI ---
MODEL_DIR_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model" 
DEFAULT_JSON_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/dude_paths_prepared_arf.json"

if MODEL_DIR_PATH not in sys.path: 
    sys.path.append(MODEL_DIR_PATH)

try:
    # 🎯 ESM OLMAYAN STANDART MODEL IMPORT EDİLİYOR
    from PLANET_model import PLANET
    print("✅ Standart PLANET_model YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# CHEMUTILS & RESIDUE ENCODING
# ==============================================================================
MAX_NB = 10
LIGAND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
ATOM_FDIM = len(LIGAND_ELEMENTS) + 6 + 5 + 4 + 4 + 1 
BOND_FDIM = 6 
AA_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return [move_to_device(t, device) for t in data]
    return data

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGAND_ELEMENTS) + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3]) + onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other']) + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(), bond.GetIsConjugated()])

def mass_center_from_pdb(pdb_lines):
    coords = []
    alphas = []
    for line in pdb_lines:
        try:
            c = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            coords.append(c)
            if line[12:16].strip() == "CA": alphas.append(c)
        except: continue
    coords = np.array(coords, dtype=np.float32)
    if len(alphas) > 0: alphas = np.array(alphas, dtype=np.float32)
    else: alphas = np.mean(coords, axis=0, keepdims=True) if len(coords)>0 else np.zeros((1,3))
    center = np.mean(coords, axis=0, keepdims=True) if len(coords)>0 else np.zeros((1,3))
    return center, coords, alphas

def get_residue_feature(res_name_3, feature_dim=20):
    """Residue isminden özellik oluşturur (ESM yerine)."""
    res_1 = AA_3_TO_1.get(res_name_3, 'X')
    vec = np.zeros(feature_dim, dtype=np.float32)
    if res_1 in AA_LIST:
        idx = AA_LIST.index(res_1)
        vec[idx] = 1.0 # One-hot başlangıç (Modelin embedding katmanı gerisini halleder)
    return vec

def mol_batch_to_graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds = [], [padding] 
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
    
    if len(fatoms) == 0: return None
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    total_bonds = len(all_bonds)
    agraph, bgraph = torch.zeros(total_atoms, MAX_NB).long(), torch.zeros(total_bonds, MAX_NB).long()
    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]): 
            if i < MAX_NB: agraph[a,i] = b
    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if i < MAX_NB and all_bonds[b2][0] != y: bgraph[b1,i] = b2
    return (fatoms, fbonds, agraph, bgraph, lig_scope)

# ==============================================================================
# DATA CLASSES
# ==============================================================================
class Residue():
    def __init__(self, residue_content, res_name_3, feature_dim): 
        self.mass_center, self.coordinates, self.alpha_position = mass_center_from_pdb(residue_content)
        self.feature = get_residue_feature(res_name_3, feature_dim)

class PocketProcessor():
    def __init__(self, protein_pdb, crystal_ligand_path, feature_dim):
        self.fresidues = []
        self.alpha_coordinates = []
        self.valid = False
        
        try:
            with open(protein_pdb,'r') as f: 
                pdb_content = [line.strip() for line in f if line.startswith('ATOM')]
            
            # Referans Ligand Okuma
            if crystal_ligand_path.endswith('.mol2'):
                ref_mol = Chem.MolFromMol2File(crystal_ligand_path, sanitize=False)
            else:
                ref_mol = Chem.SDMolSupplier(crystal_ligand_path, sanitize=False)[0]
            
            conf = ref_mol.GetConformer()
            lig_coords = np.array([conf.GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())])
            center = np.mean(lig_coords, axis=0)

            # Cep Belirleme (12 Angstrom)
            coords_list = []
            lines_list = []
            for line in pdb_content:
                c = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                coords_list.append(c); lines_list.append(line)
            
            dists = np.sum((np.array(coords_list) - center)**2, axis=1)
            pocket_lines = [lines_list[i] for i in range(len(lines_list)) if dists[i] <= 144.0]

            res_dict = defaultdict(list)
            res_names = {}
            for line in pocket_lines:
                match = re.search(r'\d+', line[22:26].strip())
                if not match: continue
                key = (int(match.group()), line[21].strip() or 'A')
                res_dict[key].append(line); res_names[key] = line[17:20].strip()

            for key, lines in res_dict.items():
                res = Residue(lines, res_names[key], feature_dim)
                self.fresidues.append(res.feature)
                pos = res.alpha_position if res.alpha_position.shape[0] > 0 else res.mass_center
                self.alpha_coordinates.append(pos.reshape(1,3))

            if len(self.fresidues) > 0:
                self.fresidues = torch.from_numpy(np.array(self.fresidues)).float()
                self.alpha_coordinates = torch.from_numpy(np.concatenate(self.alpha_coordinates, axis=0)).float()
                self.valid = True
        except: self.valid = False

class LigandDataset(Dataset):
    def __init__(self, sdf_file, max_limit=None):
        self.mols = []
        try:
            with gzip.open(sdf_file, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False)
                for mol in suppl:
                    if mol: self.mols.append(mol)
        except: pass
        if max_limit and len(self.mols) > max_limit:
            random.seed(42); self.mols = random.sample(self.mols, max_limit)

    def __len__(self): return len(self.mols)
    def __getitem__(self, idx): return self.mols[idx]

def calculate_ef_metrics(active_scores, decoy_scores):
    y_true = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_score = np.concatenate([active_scores, decoy_scores])
    sorted_indices = np.argsort(-y_score)
    sorted_labels = y_true[sorted_indices]
    n_total, n_actives = len(y_score), len(active_scores)
    ef_results = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n_sample = math.ceil(n_total * pct)
        if n_sample == 0: ef_results[f'EF_{pct*100}%'] = 0.0; continue
        ef = (sorted_labels[:n_sample].sum() / n_sample) / (n_actives / n_total)
        ef_results[f'EF_{pct*100}%'] = ef
    return ef_results

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('--out_name', type=str, default='benchmark_NoESM')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    wandb.init(project="PLANET_Eval_NoESM", name=f"Run_{args.out_name}", config=args)

    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, str(device))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    with open(DEFAULT_JSON_PATH, 'r') as f: data_map = json.load(f)
    csv_path = os.path.join(args.out_dir, f'{args.out_name}.csv')
    
    completed_targets, results = set(), []
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        results = existing_df.to_dict('records')
        completed_targets = set(existing_df['Target'].tolist())

    for target_name, records in data_map.items():
        if not records or target_name in completed_targets: continue
        
        print(f"\n--- {target_name} İşleniyor ---")
        record = records[0]
        # Doğru olan satır (feature_dim değerini 20 olarak sabitle):
        pocket_data = PocketProcessor(record['protein_pdb'], record['ligand_mol2'], 20)
        
        if not pocket_data.valid: continue
        p_fresidues = pocket_data.fresidues.to(device)
        p_coords = pocket_data.alpha_coordinates.to(device)

        def predict(loader):
            scores = []
            with torch.no_grad():
                for batch in loader:
                    if batch is None: continue
                    fatoms, fbonds, agraph, bgraph, lig_scope = [move_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in batch]
                    batch_size = len(lig_scope)
                    res_scope = [(i*p_fresidues.shape[0], p_fresidues.shape[0]) for i in range(batch_size)]
                    
                    # 🎯 Model Çağrısı
                    _, _, affinities = model((p_fresidues.repeat(batch_size, 1), None, res_scope, p_coords.repeat(batch_size, 1)), 
                                             (fatoms, fbonds, agraph, bgraph, lig_scope))
                    scores.append(affinities.cpu().numpy().flatten())
            return np.concatenate(scores) if scores else []

        act_ds = LigandDataset(record['actives_file'])
        dec_ds = LigandDataset(record['decoys_file'])
        
        act_scores = predict(DataLoader(act_ds, batch_size=args.batch_size, collate_fn=mol_batch_to_graph))
        dec_scores = predict(DataLoader(dec_ds, batch_size=args.batch_size, collate_fn=mol_batch_to_graph))

        if len(act_scores) > 0 and len(dec_scores) > 0:
            y_true = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
            y_score = np.concatenate([act_scores, dec_scores])
            roc_auc = roc_auc_score(y_true, y_score)
            ef_metrics = calculate_ef_metrics(act_scores, dec_scores)
            
            log_data = {"Target": target_name, "ROC_AUC": roc_auc}
            log_data.update(ef_metrics)
            wandb.log(log_data)
            results.append(log_data)
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"🎯 AUC: {roc_auc:.4f} | EF_1%: {ef_metrics['EF_1.0%']:.2f}")

    wandb.finish()