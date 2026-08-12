import torch
import numpy as np
import argparse, os, math, sys
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import Dataset, DataLoader
import rdkit.Chem as Chem
from tqdm import tqdm
import json
import gzip
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- PATH SETUP ---
PLANET_MODEL_DIR = os.path.join(PLANET_ROOT, "Model")
DEFAULT_JSON_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_paths_prepared_arf.json")

if PLANET_MODEL_DIR not in sys.path:
    sys.path.append(PLANET_MODEL_DIR)

try:
    from PLANET_model import PLANET
    from chemutils import mol_batch_to_graph
except ImportError as e:
    print(f"Hata: Model veya utils yüklenemedi: {e}")
    sys.exit(1)

# ==============================================================================
# 🧬 BLOSUM62 MATRIX (20-DIM FEATURE)
# ==============================================================================
BLOSUM62 = {
    'ALA':np.array([[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0]],dtype=np.float32),
    'ARG':np.array([[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3]],dtype=np.float32),
    'ASN':np.array([[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3]],dtype=np.float32),
    'ASP':np.array([[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3]],dtype=np.float32),
    'CYS':np.array([[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]],dtype=np.float32),
    'GLN':np.array([[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2]],dtype=np.float32),
    'GLU':np.array([[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'GLY':np.array([[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3]],dtype=np.float32),
    'HIS':np.array([[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3]],dtype=np.float32),
    'ILE':np.array([[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3]],dtype=np.float32),
    'LEU':np.array([[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]],dtype=np.float32),
    'LYS':np.array([[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'MET':np.array([[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1]],dtype=np.float32),
    'PHE':np.array([[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1]],dtype=np.float32),
    'PRO':np.array([[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2]],dtype=np.float32),
    'SER':np.array([[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2]],dtype=np.float32),
    'THR':np.array([[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0]],dtype=np.float32),
    'TRP':np.array([[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3]],dtype=np.float32),
    'TYR':np.array([[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1]],dtype=np.float32),
    'VAL':np.array([[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]],dtype=np.float32)
}

# ==============================================================================
# 🛠️ ROBUST POCKET PROCESSOR
# ==============================================================================
class RobustPocketProcessor:
    def __init__(self, pdb_path, ligand_path):
        self.res_features = []
        self.alpha_coordinates = []
        self.valid = False

        # 1. Ligand Center
        try:
            mol = None
            if ligand_path.endswith('.mol2'):
                mol = Chem.MolFromMol2File(ligand_path, sanitize=False)
            else:
                suppl = Chem.SDMolSupplier(ligand_path, sanitize=False)
                if len(suppl) > 0: mol = suppl[0]
            
            if mol is None: return 

            conf = mol.GetConformer()
            lig_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            center = np.mean(lig_coords, axis=0)
        except: return

        # 2. PDB Parsing
        try:
            with open(pdb_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    if res_name == 'MSE': res_name = 'MET'
                    
                    if res_name in BLOSUM62:
                        try:
                            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                            pos = np.array([x, y, z])
                            
                            if np.linalg.norm(pos - center) <= 12.0:
                                self.alpha_coordinates.append(pos)
                                self.res_features.append(BLOSUM62[res_name].flatten())
                        except: pass
            
            if len(self.alpha_coordinates) > 0:
                self.valid = True

        except Exception as e:
            return

# ==============================================================================
# DATASET & METRICS
# ==============================================================================
class VirtualScreening_Dataset(Dataset):
    def __init__(self, ligand_file):
        self.mols = []
        if ligand_file.endswith('.sdf') or ligand_file.endswith('.sdf.gz'):
            import gzip
            open_func = gzip.open if ligand_file.endswith('.gz') else open
            try:
                with open_func(ligand_file, 'rb') as f:
                    suppl = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False)
                    for mol in suppl:
                        if mol: self.mols.append(mol)
            except: pass
        else: # SMILES
            try:
                with open(ligand_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            smi = line.strip().split()[0]
                            m = Chem.MolFromSmiles(smi)
                            if m: self.mols.append(Chem.AddHs(m))
            except: pass

    def __len__(self): return len(self.mols)
    def __getitem__(self, idx): return self.mols[idx]

def collate_fn(batch):
    valid_batch = [m for m in batch if m is not None]
    if not valid_batch: return None
    return mol_batch_to_graph(valid_batch)

def calculate_metrics(act_scores, dec_scores):
    if len(act_scores) == 0 or len(dec_scores) == 0: return {}
    
    y_true = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
    y_score = np.concatenate([act_scores, dec_scores])
    
    metrics = {}
    try:
        # ROC & PR
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['PR_AUC'] = auc(recall, precision)
        
        # EF Calculation
        sorted_indices = np.argsort(-y_score)
        sorted_labels = y_true[sorted_indices]
        n_total = len(y_score)
        n_actives = len(act_scores)
        
        # EF Thresholds: 0.5%, 1%, 2%, 5%
        for pct in [0.005, 0.01, 0.02, 0.05]:
            n_sample = math.ceil(n_total * pct)
            if n_sample == 0: 
                metrics[f'EF_{pct*100}%'] = 0.0
                continue
                
            hits = sorted_labels[:n_sample].sum()
            random_rate = n_actives / n_total
            found_rate = hits / n_sample
            
            ef = found_rate / random_rate if random_rate > 0 else 0.0
            metrics[f'EF_{pct*100}%'] = ef
            
    except: pass
    return metrics

# ==============================================================================
# PREDICT
# ==============================================================================
def predict(model, protein_pdb, ligand_ref, dataloader, device):
    pocket = RobustPocketProcessor(protein_pdb, ligand_ref)
    
    if not pocket.valid or len(pocket.alpha_coordinates) == 0:
        return np.array([])

    try:
        # 20 dim features (No Padding needed, model projects it)
        res_features_np = np.array(pocket.res_features, dtype=np.float32) 
        coords_np = np.array(pocket.alpha_coordinates, dtype=np.float32)
        
        res_features = torch.from_numpy(res_features_np).to(device)
        alpha_coords = torch.from_numpy(coords_np).to(device)
        
        # Model context preparation
        fresidues = model.cal_res_features_helper(res_features, alpha_coords)
        
    except: return np.array([])

    predicted_affinities = []
    model.eval()
    
    with torch.no_grad():
        for mol_feature_batch in dataloader:
            if mol_feature_batch is None: continue

            fatoms, fbonds, agraph, bgraph, scope = mol_feature_batch
            mol_feature_batch_gpu = (
                fatoms.to(device), fbonds.to(device), agraph.to(device), 
                bgraph.to(device), scope
            )

            try:
                batch_size = len(scope)
                fresidues_batch, res_scope = model.cal_res_features(fresidues, batch_size)
                prediction = model.screening(fresidues_batch, res_scope, mol_feature_batch_gpu)
                predicted_affinities.append(prediction.view(-1).cpu().numpy())
            except: continue

    if not predicted_affinities: return np.array([])
    return np.concatenate(predicted_affinities, axis=0)

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--model_file', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-d', '--dude_dir', type=str, default='') 
    
    # Model Params (Must correspond to training config)
    parser.add_argument('--feature_dims', type=int, default=300) 
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--key_dims', type=int, default=300)
    parser.add_argument('--value_dims', type=int, default=300)
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Load Model
    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims, 3, 10, 1, str(device)).to(device)
    try:
        model.load_state_dict(torch.load(args.model_file, map_location=device))
        print("✅ Model Weights Loaded.")
    except Exception as e:
        sys.exit(f"❌ Model Load Error: {e}")
    
    model.eval()

    if not os.path.exists(DEFAULT_JSON_PATH): sys.exit("JSON not found")
    with open(DEFAULT_JSON_PATH, 'r') as f: data_map = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    stats = []
    csv_path = os.path.join(args.out_dir, 'benchmark_original_planet_full.csv')
    
    print(f"📂 Processing {len(data_map)} targets...")

    for target_name, target_info in tqdm(data_map.items()):
        if not target_info: continue
        info = target_info[0]
        
        protein_pdb = info['protein_pdb']
        ligand_ref = info['ligand_mol2'] 
        actives_file = info['actives_file']
        decoys_file = info['decoys_file']
        
        if not (os.path.exists(protein_pdb) and os.path.exists(ligand_ref)): continue

        # Data Loaders
        act_loader = DataLoader(VirtualScreening_Dataset(actives_file), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
        dec_loader = DataLoader(VirtualScreening_Dataset(decoys_file), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        # Predictions
        act_scores = predict(model, protein_pdb, ligand_ref, act_loader, device)
        dec_scores = predict(model, protein_pdb, ligand_ref, dec_loader, device)

        # Calculate Metrics
        metrics = calculate_metrics(act_scores, dec_scores)
        
        if metrics:
            metrics['Target'] = target_name
            stats.append(metrics)
            
            # Print current target result
            print(f"   🎯 {target_name} -> AUC: {metrics['ROC_AUC']:.4f} | EF 1%: {metrics.get('EF_1.0%', 0):.2f}")
            
            # Save intermediate results
            pd.DataFrame(stats).to_csv(csv_path, index=False)
        
        if device.type == 'cuda': torch.cuda.empty_cache()

    print(f"\n✅ Completed. Saved to: {csv_path}")