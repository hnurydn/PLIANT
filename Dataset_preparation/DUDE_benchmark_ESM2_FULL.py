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
import traceback
import pickle
import random
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- PATH AYARLARI ---
MODEL_DIR_PATH = os.path.join(PLANET_ROOT, "Model") 
JSON_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_paths_prepared_arf.json")
PCA_PARAMS_PATH = os.path.join(PLANET_ROOT, "esm2_pca_params.pkl") # PCA Dosyası

if MODEL_DIR_PATH not in sys.path: sys.path.append(MODEL_DIR_PATH)

try:
    from PLANET_model_esm2 import PLANET
    print("✅ PLANET_model_esm2 YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model_esm2.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# AYARLAR & CHEMUTILS
# ==============================================================================
MAX_NB = 10
FEATURE_DIM = 512 
periodic_table = Chem.GetPeriodicTable()
PROTEIN_ELEMENTS = ['H','C','N','O','S','P','Se']
LIGADND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1 
BOND_FDIM = 6  
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

# --- GLOBAL PCA YÜKLEME ---
PCA_MEAN = None
PCA_TRANSFORM = None

def load_pca_params():
    global PCA_MEAN, PCA_TRANSFORM
    if os.path.exists(PCA_PARAMS_PATH):
        try:
            with open(PCA_PARAMS_PATH, 'rb') as f:
                params = pickle.load(f)
                if isinstance(params, dict):
                    PCA_MEAN = params['mean']
                    PCA_TRANSFORM = params['components']
                else: # Sklearn objesi
                    PCA_MEAN = params.mean_
                    PCA_TRANSFORM = params.components_
            print("✅ PCA Parametreleri yüklendi.")
        except Exception as e:
            print(f"❌ PCA Yükleme Hatası: {e}")

load_pca_params()

def apply_pca_transform_single(embedding_1280d):
    """ Tek bir vektöre PCA uygular (Eğitim koduyla aynı mantık) """
    if PCA_TRANSFORM is None: return embedding_1280d[:512] # Fallback
    # Merkezleme ve Dönüşüm
    embedding_centered = embedding_1280d - PCA_MEAN
    embedding_pca = np.dot(embedding_centered, PCA_TRANSFORM.T)
    return embedding_pca.astype(np.float32)

# --- CHEMUTILS FONKSİYONLARI ---
def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGADND_ELEMENTS) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other'])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
        bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)

def near_pocket(pdb_line,centeroid):
    coordinates = pdb_line[30:55]
    x, y, z = float(coordinates[0:8]), float(coordinates[8:16]), float(coordinates[16:24])
    return np.sqrt(np.sum(np.square(np.array([x,y,z],dtype=np.float32)-centeroid))) <= 12.0

def mass_center_from_pdb(pdb_lines):
    pdb_lines = [line for line in pdb_lines if line[76:78].strip().capitalize() in PROTEIN_ELEMENTS]
    if not pdb_lines: return np.zeros((1,3)), np.zeros((0,3)), np.zeros((0,3))
    atoms_mass = np.reshape(np.array([periodic_table.GetAtomicWeight(line[76:78].strip().capitalize()) for line in pdb_lines]),[-1,1])
    coordinates = np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pdb_lines],dtype=np.float32)
    if np.sum(atoms_mass) == 0: mass_center = np.mean(coordinates, axis=0, keepdims=True)
    else: mass_center = np.sum(coordinates * atoms_mass,axis=0,keepdims=True) / np.sum(atoms_mass)
    alpha_position = np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pdb_lines if line[12:16].strip()=="CA"],dtype=np.float32)
    if alpha_position.shape[0] == 0: alpha_position = mass_center
    return mass_center, coordinates, alpha_position

# --- KRİTİK FONKSİYON: EĞİTİM KODUNDAN ALINDI ---
def parse_fasta_from_pdb_content(pdb_content):
    """ PDB içeriğinden mapping oluşturur. """
    sequences = {}
    chain_res_info = {}
    chain_order = []
    current_res_num = None
    
    for line in pdb_content:
        if line.startswith("ATOM"):
            chain_id = line[21].strip() or 'A'
            res_num_str = line[22:26].strip()
            res_type_3 = line[17:20].strip()
            
            if not res_num_str or res_type_3 not in AA_3_TO_1: continue
            
            res_id_key = (chain_id, int(res_num_str))
            
            if chain_id not in sequences:
                sequences[chain_id] = []
                chain_res_info[chain_id] = []
                chain_order.append(chain_id)
            
            if res_id_key != current_res_num:
                one_letter = AA_3_TO_1.get(res_type_3, '')
                if one_letter:
                    sequences[chain_id].append(one_letter)
                    chain_res_info[chain_id].append(res_id_key)
                    current_res_num = res_id_key
                    
    mapping = {}
    global_index = 0
    for chain in chain_order:
        for res_key in chain_res_info[chain]:
            mapping[res_key] = global_index
            global_index += 1
    return mapping

class Residue():
    def __init__(self, residue_content, res_num_str, chain_id): 
        self.mass_center, self.coordinates, self.alpha_position = mass_center_from_pdb(residue_content)
        self.residue_type = residue_content[0][17:20].replace('MSE','MET')
        self.feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.res_num = int(res_num_str)
        self.chain_id = chain_id

    def get_feature(self): return self.feature
    def get_alpha_position(self): return self.alpha_position
    def get_mass_center(self): return self.mass_center

class Mol():
    def __init__(self, mol):
        self.mol = mol
        self.atom_count = mol.GetNumAtoms()
    def get_atom_coordinates(self):
        conf = self.mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in range(self.atom_count)], dtype=np.float32)
    def compute_centeroid(self): return np.mean(self.get_atom_coordinates(), axis=0)

class ComplexPocket():
    def __init__(self, protein_pdb, ligand_sdf, pK=0, esm_embedding=None):
        with open(protein_pdb,'r') as pdb_file:
            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')]
        
        # --- 1. MAPPING OLUŞTUR (Eğitimdeki gibi) ---
        mapping = parse_fasta_from_pdb_content(pdb_content)
        
        self.ligand = Mol(ligand_sdf) if not isinstance(ligand_sdf, Mol) else ligand_sdf
        self.pK = pK
        centeroid = self.ligand.compute_centeroid()
        self.pocket_residues = []
        
        pocket_atoms_content = [line for line in pdb_content if near_pocket(line, centeroid)]
        res_dict = defaultdict(list)
        for line in pocket_atoms_content:
            res_dict[(line[22:27].strip(), line[21])].append(line)
        
        # --- 2. RESIDUE OLUŞTUR VE EŞLEŞTİR ---
        for (res_num_str, chain), lines in res_dict.items():
            try:
                res = Residue(lines, res_num_str, chain)
                
                if esm_embedding is not None:
                    chain_id = chain if chain.strip() else 'A'
                    key = (chain_id, int(res_num_str))
                    
                    if key in mapping:
                        idx = mapping[key]
                        if idx < len(esm_embedding):
                            # Ham 1280 vektörü al
                            raw_vec = esm_embedding[idx]
                            # PCA Uygula (Eğitimdeki kodun aynısı)
                            pca_vec = apply_pca_transform_single(raw_vec)
                            res.feature = pca_vec
                        else:
                            # Index taşarsa
                            pass
                    else:
                        # Mapping'de yoksa
                        pass

                self.pocket_residues.append(res)
            except: continue
        self.res_count = len(self.pocket_residues)

def tensorize_protein_pocket(protein_pockets):
    fresidues, res_scope, alpha_coordinates = [], [], []
    total_residues = 0
    res_map = [] 
    for pocket in protein_pockets:
        residue_count = len(pocket.pocket_residues)
        for residue in pocket.pocket_residues:
            fresidues.append(residue.get_feature())
            pos = residue.get_alpha_position()
            if pos is None or len(pos) == 0 or (hasattr(pos, 'shape') and pos.shape[0] == 0): pos = residue.get_mass_center()
            if pos.ndim == 1: pos = pos.reshape(1, 3)
            alpha_coordinates.append(pos)
        res_scope.append((total_residues, residue_count))
        total_residues += residue_count

    if len(fresidues) > 0:
        fresidues = torch.from_numpy(np.concatenate(fresidues, axis=0).reshape(-1, FEATURE_DIM))
        alpha_coordinates = torch.from_numpy(np.concatenate(alpha_coordinates, axis=0).astype(np.float32))
        res_map = torch.zeros((total_residues, total_residues, 1))
        for (start, count) in res_scope: res_map[start:start+count, start:start+count, :] = 1.0
    else:
        fresidues = torch.zeros((0, FEATURE_DIM)); alpha_coordinates = torch.zeros((0, 3)); res_map = torch.zeros((0, 0, 1))
    return (fresidues, res_map, res_scope, alpha_coordinates)

def mol_batch_to_graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds = [], [padding] 
    in_bonds, all_bonds = [], [(-1,-1)] 
    lig_scope, total_atoms = [], 0
    for mol in mol_batch:
        try: Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        except: pass
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms(): fatoms.append(atom_features(atom)); in_bonds.append([])
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx() + total_atoms, bond.GetEndAtomIdx() + total_atoms
            b = len(all_bonds); all_bonds.append((a1,a2)); fbonds.append(torch.cat([fatoms[a1], bond_features(bond)], 0)); in_bonds[a2].append(b)
            b = len(all_bonds); all_bonds.append((a2,a1)); fbonds.append(torch.cat([fatoms[a2], bond_features(bond)], 0)); in_bonds[a1].append(b)
        lig_scope.append((total_atoms, n_atoms)); total_atoms += n_atoms

    if len(fatoms) > 0: fatoms = torch.stack(fatoms, 0)
    else: fatoms = torch.zeros((0, ATOM_FDIM + BOND_FDIM))
    if len(fbonds) > 1: fbonds = torch.stack(fbonds, 0) 
    else: fbonds = torch.stack([padding], 0)
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

def tensorize_ligand(mol_batch):
    graph = mol_batch_to_graph([m.mol for m in mol_batch])
    mol_int = torch.zeros(sum([m.atom_count for m in mol_batch]))
    return graph, mol_int

def tensorize_all(protein_pockets, decoy_flag=True):
    res_pack = tensorize_protein_pocket(protein_pockets)
    mol_pack, mol_int = tensorize_ligand([p.ligand for p in protein_pockets])
    return res_pack, mol_pack, mol_int, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

# ==============================================================================
# MAIN & DATA LOADER
# ==============================================================================
def load_sdf_gz_iterator(filename):
    try:
        with gzip.open(filename, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False)
            for mol in suppl:
                if mol is not None:
                    try: Chem.SanitizeMol(mol)
                    except: pass
                    yield mol
    except: pass

"""class ESMDataset(Dataset):
    def __init__(self, sdf_path, protein_path, esm_embedding):
        self.protein_path = protein_path
        self.esm_embedding = esm_embedding 
        print(f"  ⏳ Okunuyor: {os.path.basename(sdf_path)}...", flush=True)
        self.mols = list(load_sdf_gz_iterator(sdf_path))
        print(f"  ✅ {len(self.mols)} molekül yüklendi.", flush=True)"""

class ESMDataset(Dataset):
    def __init__(self, sdf_path, protein_path, esm_embedding, max_limit=None): # max_limit parametresi eklendi
        self.protein_path = protein_path
        self.esm_embedding = esm_embedding 
        print(f"  ⏳ Okunuyor: {os.path.basename(sdf_path)}...", flush=True)
        
        # Tüm molekülleri yükle
        self.mols = list(load_sdf_gz_iterator(sdf_path))
        total_mols = len(self.mols)
        
        # --- RASTGELE AZALTMA (SAMPLING) ---
        if max_limit is not None and total_mols > max_limit:
            print(f"  ✂️ {total_mols} molekülden rastgele {max_limit} tanesi seçiliyor...", flush=True)
            random.seed(42) # Sonuçların her seferinde aynı olması için (Reproducibility)
            self.mols = random.sample(self.mols, max_limit)
        
        print(f"  ✅ {len(self.mols)} molekül işleme alındı.", flush=True)
    def __len__(self): return len(self.mols)
    def __getitem__(self, idx):
        try:
            mol = self.mols[idx]
            pocket = ComplexPocket(self.protein_path, mol, pK=0.0, esm_embedding=self.esm_embedding)
            if not pocket.pocket_residues or pocket.ligand.atom_count == 0: return None
            return pocket
        except: return None

skipped_count = 0
def safe_collate_fn(batch_list):
    global skipped_count
    valid_batch = []
    for item in batch_list:
        if item is None: skipped_count += 1; continue
        try:
            valid_res = 0
            for res in item.pocket_residues:
                if res.get_alpha_position() is not None: valid_res += 1
            if valid_res > 0: valid_batch.append(item)
            else: skipped_count += 1
        except: skipped_count += 1; continue
    if len(valid_batch) == 0: return None
    try: return tensorize_all(valid_batch, decoy_flag=False)
    except: skipped_count += len(valid_batch); return None

def run_inference(model, dataloader, device):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if batch_data is None: continue
            try:
                res_feature_batch = batch_data[0]
                mol_feature_batch = batch_data[1]
                
                fresidues, res_map, res_scope, alpha_coordinates = res_feature_batch
                if fresidues.shape[0] == 0: continue

                fresidues = fresidues.to(device)
                alpha_coordinates = alpha_coordinates.to(device)
                
                fatoms, fbonds, agraph, bgraph, lig_scope = mol_feature_batch
                fatoms = fatoms.to(device)
                fbonds = fbonds.to(device)
                agraph = agraph.to(device)
                bgraph = bgraph.to(device)
                
                res_inputs = (fresidues, res_map, res_scope, alpha_coordinates)
                mol_inputs = (fatoms, fbonds, agraph, bgraph, lig_scope)
                
                _, _, predicted_affinities = model(res_inputs, mol_inputs)
                all_scores.append(predicted_affinities.cpu().numpy().flatten())
            except Exception as e:
                continue
    if not all_scores: return np.array([])
    return np.concatenate(all_scores)

def calculate_metrics(active_scores, decoy_scores):
    if len(active_scores) == 0 or len(decoy_scores) == 0: return None
    y_true = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_scores = np.concatenate([active_scores, decoy_scores])
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        sorted_indices = np.argsort(-y_scores)
        sorted_labels = y_true[sorted_indices]
        n_total = len(y_scores)
        n_actives = len(active_scores)
        ef_results = {}
        for percentage in [0.005, 0.01, 0.02, 0.05]:
            n_sample = math.ceil(n_total * percentage)
            if n_sample == 0: 
                ef_results[f'EF_{percentage*100}%'] = 0.0
                continue
            n_actives_found = sorted_labels[:n_sample].sum()
            ef = (n_actives_found / n_sample) / (n_actives / n_total) if n_actives > 0 else 0.0
            ef_results[f'EF_{percentage*100}%'] = ef
            
        return roc_auc, pr_auc, ef_results
    except: return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('-h5', '--h5_path', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-s','--feature_dims', type=int, default=512)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=128)
    parser.add_argument('-va','--value_dims', type=int, default=128)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)

    args = parser.parse_args()
    
    wandb.init(project="PLANET_Benchmark_ESM2_FINAL_FIXED", name=f"Eval_FIXED_{os.path.basename(args.model_path)}", config=args)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, str(device))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    with open(JSON_PATH, 'r') as f: data_map = json.load(f)
    print(f"🚀 {len(data_map)} Hedef JSON dosyasından okundu.")
    results = []
    csv_path = os.path.join(args.out_dir, 'benchmark_results_esm2_final_fixed.csv')
    
    if not os.path.exists(args.h5_path): sys.exit("❌ H5 Yok")
    h5_file = h5py.File(args.h5_path, 'r')
    print("✅ H5 Embeddings yüklendi.")

    for target_name, records in data_map.items():
        if not records: continue
        
        # --- TEST İÇİN SADECE ADRB2'Yİ AÇ, SONRA HEPSİNİ AÇARSIN ---
        # if target_name != "adrb2": continue 
        
        record = records[0] 
        print(f"\n--- İşleniyor (FIXED): {target_name} ---", flush=True)
        
        target_esm = None
        if target_name in h5_file: target_esm = h5_file[target_name][:]
        else: print(f"⚠️ UYARI: {target_name} için embedding yok!")

        skipped_count = 0
        
        """try:
            active_ds = ESMDataset(record['actives_file'], record['protein_pdb'], target_esm)
            decoy_ds = ESMDataset(record['decoys_file'], record['protein_pdb'], target_esm)"""
            
        try:
            # Active'leri HEPSİNİ al (Onlar zaten az, genelde 100-500 arası)
            active_ds = ESMDataset(record['actives_file'], record['protein_pdb'], target_esm, max_limit=None)
            
            # Decoy'ları LİMİTLE (Örneğin en fazla 1000 tane al)
            # 1000 - 2000 arası idealdir. Ne kadar çok o kadar iyi ama 1000 de fikir verir.
            decoy_ds = ESMDataset(record['decoys_file'], record['protein_pdb'], target_esm, max_limit=None)
            
            if len(active_ds) == 0 or len(decoy_ds) == 0: continue

            active_loader = DataLoader(active_ds, batch_size=args.batch_size, shuffle=False, 
                                       collate_fn=safe_collate_fn, num_workers=10)
            decoy_loader = DataLoader(decoy_ds, batch_size=args.batch_size, shuffle=False, 
                                      collate_fn=safe_collate_fn, num_workers=10)
            
            print(f"Actives ({len(active_ds)}) tahmin ediliyor...", flush=True)
            active_scores = run_inference(model, active_loader, device)
            print(f"Decoys ({len(decoy_ds)}) tahmin ediliyor...", flush=True)
            decoy_scores = run_inference(model, decoy_loader, device)
            
            print(f"Stats: Skip={skipped_count} | Act={len(active_scores)} | Dec={len(decoy_scores)}", flush=True)

            metrics = calculate_metrics(active_scores, decoy_scores)
            if metrics:
                roc_auc, pr_auc, ef_dict = metrics
                print(f"🎯 {target_name} -> ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}", flush=True)
                log_data = {"Target": target_name, "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "Skipped": skipped_count}
                log_data.update(ef_dict)
                wandb.log(log_data)
                results.append(log_data)
                pd.DataFrame(results).to_csv(csv_path, index=False)
                print(f"💾 KAYDEDİLDİ: {csv_path}", flush=True)
        except Exception as e:
            print(f"❌ HATA ({target_name}): {e}", flush=True)

    h5_file.close()
    print(f"\n✅ BİTTİ! Dosya: {csv_path}")
    wandb.finish()