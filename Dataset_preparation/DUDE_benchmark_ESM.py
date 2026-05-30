import torch
import torch.nn as nn
import numpy as np
import argparse, os, sys, json, gzip, pandas as pd, math, wandb, random, re, h5py, pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import Chem, RDLogger
from collections import defaultdict

# --- PATH AYARLARI ---
MODEL_DIR_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model" 
DEFAULT_JSON_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/dude_paths_prepared_arf.json"
PCA_PARAMS_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/esm2_pca_params.pkl"

if MODEL_DIR_PATH not in sys.path: sys.path.append(MODEL_DIR_PATH)

try:
    from PLANET_model_esm2 import PLANET
    print("✅ PLANET_model_esm2 YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model_esm2.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# CHEMUTILS
# ==============================================================================
MAX_NB = 10
FEATURE_DIM = 512 
LIGADND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1 
BOND_FDIM = 6  
AA_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

# --- PCA LOAD ---
PCA_MEAN = None
PCA_TRANSFORM = None
def load_pca_params():
    global PCA_MEAN, PCA_TRANSFORM
    if os.path.exists(PCA_PARAMS_PATH):
        try:
            with open(PCA_PARAMS_PATH, 'rb') as f:
                params = pickle.load(f)
                if isinstance(params, dict):
                    PCA_MEAN = params['mean']; PCA_TRANSFORM = params['components']
                else: 
                    PCA_MEAN = params.mean_; PCA_TRANSFORM = params.components_
            print("✅ PCA Parametreleri yüklendi.")
        except: pass
load_pca_params()

def apply_pca_transform_single(embedding_1280d):
    if PCA_TRANSFORM is None: return embedding_1280d[:512]
    emb = embedding_1280d - PCA_MEAN
    return np.dot(emb, PCA_TRANSFORM.T).astype(np.float32)

# --- HELPER FUNCTIONS ---
def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGADND_ELEMENTS) + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3]) + onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other']) + [atom.GetIsAromatic()])

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

def parse_fasta_from_pdb_content(pdb_content):
    mapping = {}
    idx = 0
    seen = set()
    for line in pdb_content:
        if line.startswith("ATOM"):
            res_id_str = line[22:26].strip()
            match = re.search(r'\d+', res_id_str)
            if not match: continue
            res_id = int(match.group())
            chain = line[21].strip() or 'A'
            key = (chain, res_id)
            if key not in seen and line[17:20].strip() in AA_3_TO_1:
                mapping[key] = idx
                seen.add(key)
                idx += 1
    return mapping

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

# ==============================================================================
# CLASSES
# ==============================================================================

class Residue():
    def __init__(self, residue_content, res_num_val, chain_id): 
        self.mass_center, self.coordinates, self.alpha_position = mass_center_from_pdb(residue_content)
        self.feature = np.zeros(FEATURE_DIM, dtype=np.float32)

class PocketProcessor():
    def __init__(self, protein_pdb, crystal_ligand_path, esm_embedding):
        self.fresidues = []
        self.alpha_coordinates = []
        self.valid = False
        
        try:
            with open(protein_pdb,'r') as f: 
                pdb_content = [line.strip() for line in f if line.startswith('ATOM')]
        except: return 
        
        ligand = None
        try:
            if crystal_ligand_path.endswith('.mol2'):
                mol = Chem.MolFromMol2File(crystal_ligand_path, sanitize=False)
                if mol: ligand = mol
            else:
                suppl = Chem.SDMolSupplier(crystal_ligand_path, sanitize=False)
                if len(suppl) > 0: ligand = suppl[0]
        except: pass
        
        if ligand is None: return
        
        try:
            conf = ligand.GetConformer()
            lig_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())])
            center = np.mean(lig_coords, axis=0)
        except: return

        coords_list = []
        lines_list = []
        for line in pdb_content:
            try:
                coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                lines_list.append(line)
            except: continue
            
        coords_np = np.array(coords_list)
        dists = np.sum((coords_np - center)**2, axis=1)
        mask = dists <= 144.0 
        
        pocket_lines = [lines_list[i] for i in range(len(lines_list)) if mask[i]]
        
        if len(pocket_lines) < 10:
            idx = np.argsort(dists)[:600]
            pocket_lines = [lines_list[i] for i in idx]

        mapping = parse_fasta_from_pdb_content(pdb_content) 
        res_dict = defaultdict(list)
        
        for line in pocket_lines:
            res_num_str = line[22:26].strip()
            match = re.search(r'\d+', res_num_str)
            if not match: continue
            res_num_val = int(match.group())
            res_dict[(res_num_val, line[21])].append(line)

        for (res_num_val, chain), lines in res_dict.items():
            res = Residue(lines, res_num_val, chain)
            if esm_embedding is not None:
                key = (chain if chain.strip() else 'A', res_num_val)
                if key in mapping:
                    idx = mapping[key]
                    if idx < len(esm_embedding):
                        res.feature = apply_pca_transform_single(esm_embedding[idx])
            
            self.fresidues.append(res.feature)
            pos = res.alpha_position
            if pos.shape[0] == 0: pos = res.mass_center 
            if pos.ndim == 1: pos = pos.reshape(1,3)
            self.alpha_coordinates.append(pos)

        if len(self.fresidues) > 0:
            self.fresidues = torch.from_numpy(np.concatenate(self.fresidues, axis=0).reshape(-1, FEATURE_DIM))
            coord_list_np = [x if x.shape == (1,3) else x.reshape(1,3) for x in self.alpha_coordinates]
            self.alpha_coordinates = torch.from_numpy(np.concatenate(coord_list_np, axis=0).astype(np.float32))
            self.valid = True
        else:
            self.valid = False

class LigandDataset(Dataset):
    def __init__(self, sdf_file, max_limit=None):
        self.mols = []
        try:
            if sdf_file.endswith('.gz'):
                with gzip.open(sdf_file, 'rb') as f:
                    suppl = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False)
                    for i, mol in enumerate(suppl):
                        if mol: self.mols.append(mol)
        except: pass
        
        if max_limit and len(self.mols) > max_limit:
            random.seed(42)
            self.mols = random.sample(self.mols, max_limit)

    def __len__(self): return len(self.mols)
    def __getitem__(self, idx): return self.mols[idx]

def collate_ligands(batch):
    valid_mols = [m for m in batch if m is not None]
    if len(valid_mols) == 0: return None
    return mol_batch_to_graph(valid_mols)

def calculate_ef_metrics(active_scores, decoy_scores):
    y_true = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_score = np.concatenate([active_scores, decoy_scores])
    sorted_indices = np.argsort(-y_score)
    sorted_labels = y_true[sorted_indices]
    n_total = len(y_score)
    n_actives = len(active_scores)
    
    ef_results = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n_sample = math.ceil(n_total * pct)
        if n_sample == 0: ef_results[f'EF_{pct*100}%'] = 0.0; continue
        n_actives_found = sorted_labels[:n_sample].sum()
        random_rate = n_actives / n_total
        found_rate = n_actives_found / n_sample
        ef = found_rate / random_rate if random_rate > 0 else 0.0
        ef_results[f'EF_{pct*100}%'] = ef
    return ef_results

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('-h5', '--h5_path', required=True)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('-s','--feature_dims', type=int, default=512)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=128)
    parser.add_argument('-va','--value_dims', type=int, default=128)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
    parser.add_argument('--out_name', type=str, default='benchmark_results')
    
    # 🎯 DÜZELTME: Bu satır eksikti, ekledik!
    parser.add_argument('--json_path', type=str, default=DEFAULT_JSON_PATH)

    # --- ARTIK KULLANILMIYOR AMA SCRIPT HATASI OLMASIN DİYE TUTUYORUZ ---
    parser.add_argument('--gating', action='store_true')
    parser.add_argument('--vector', action='store_true')

    args = parser.parse_args()
    
    if args.device: device = torch.device(args.device)
    else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu': torch.set_num_threads(os.cpu_count())

    os.makedirs(args.out_dir, exist_ok=True)
    wandb.init(project="PLANET_Eval_ESM", name=f"Run_{os.path.basename(args.model_path)}", config=args)

    print(f"⚙️ Model Yükleniyor: {args.model_path}")
    
    # Model Init (Standard)
    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, 
                   str(device))
    
    # Strict=False (Hata önleyici)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print("✅ Model ağırlıkları başarıyla yüklendi (strict=False).")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        sys.exit(1)

    model.to(device).eval()

    if not os.path.exists(args.h5_path): sys.exit("❌ H5 Yok")
    h5_file = h5py.File(args.h5_path, 'r')
    
    # 🎯 DÜZELTME: args.json_path kullanılıyor
    with open(args.json_path, 'r') as f: data_map = json.load(f)
    
    csv_path = os.path.join(args.out_dir, f'{args.out_name}.csv')
    completed_targets = set()
    results = []
    
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            results = existing_df.to_dict('records')
            completed_targets = set(existing_df['Target'].tolist())
            print(f"📂 Mevcut CSV bulundu. {len(completed_targets)} hedef tamamlanmış.")
        except: pass

    for target_name, records in data_map.items():
        if not records or target_name in completed_targets: continue
        
        record = records[0]
        print(f"\n--- {target_name} İşleniyor ---", flush=True)

        target_esm = h5_file[target_name][:] if target_name in h5_file else None
        pocket_data = PocketProcessor(record['protein_pdb'], record['ligand_mol2'], target_esm)
        
        if not pocket_data.valid:
            print("❌ Cep oluşturulamadı.")
            continue

        p_fresidues = pocket_data.fresidues.to(device)
        p_coords = pocket_data.alpha_coordinates.to(device)
        
        def predict_ligands(loader):
            scores = []
            with torch.no_grad():
                for batch in loader:
                    if batch is None: continue
                    fatoms, fbonds, agraph, bgraph, lig_scope = batch
                    data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in [fatoms, fbonds, agraph, bgraph]]
                    
                    batch_size = len(lig_scope)
                    fresidues_batch = p_fresidues.repeat(batch_size, 1)
                    coords_batch = p_coords.repeat(batch_size, 1)
                    n_res = p_fresidues.shape[0]
                    res_scope = [(i*n_res, n_res) for i in range(batch_size)]
                    
                    res_inputs = (fresidues_batch, None, res_scope, coords_batch)
                    mol_inputs = (data[0], data[1], data[2], data[3], lig_scope)
                    
                    try:
                        _, _, affinities = model(res_inputs, mol_inputs)
                        scores.append(affinities.cpu().numpy().flatten())
                    except Exception as e:
                        print(f"⚠️ Hata: {e}")
                        break
            return np.concatenate(scores) if len(scores) > 0 else []

        act_ds = LigandDataset(record['actives_file'], max_limit=None)
        dec_ds = LigandDataset(record['decoys_file'], max_limit=None) 
        act_loader = DataLoader(act_ds, batch_size=args.batch_size, collate_fn=collate_ligands, num_workers=0)
        dec_loader = DataLoader(dec_ds, batch_size=args.batch_size, collate_fn=collate_ligands, num_workers=0)
        
        if len(act_ds) > 0 and len(dec_ds) > 0:
            act_scores = predict_ligands(act_loader)
            dec_scores = predict_ligands(dec_loader)
            
            if len(act_scores) > 0 and len(dec_scores) > 0:
                y_true = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
                y_score = np.concatenate([act_scores, dec_scores])
                roc_auc = roc_auc_score(y_true, y_score)
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                ef_metrics = calculate_ef_metrics(act_scores, dec_scores)
                
                print(f"🎯 {target_name} -> AUC: {roc_auc:.4f} | EF_1%: {ef_metrics['EF_1.0%']:.2f}")
                log_data = {"Target": target_name, "ROC_AUC": roc_auc, "PR_AUC": pr_auc}
                log_data.update(ef_metrics)
                wandb.log(log_data); results.append(log_data)
                pd.DataFrame(results).to_csv(csv_path, index=False)

    h5_file.close(); wandb.finish()