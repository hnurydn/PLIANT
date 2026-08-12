import torch
import torch.nn as nn
import numpy as np
import argparse, os, sys, json, gzip, pandas as pd, math, wandb, random, re, h5py, pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from collections import defaultdict
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- 1. PROJE YOLU VE MODÜL AYARLARI ---
MODEL_DIR_PATH = os.path.join(PLANET_ROOT, "Model") 
DEFAULT_JSON_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_arf.json")
PCA_PARAMS_PATH = os.path.join(PLANET_ROOT, "esm2_pca_params.pkl")

if MODEL_DIR_PATH not in sys.path: 
    sys.path.append(MODEL_DIR_PATH)

try:
    from PLANET_model_esm2 import PLANET
    print("✅ PLANET_model_esm2 YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model_esm2.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# 2. CHEMUTILS & YARDIMCI FONKSİYONLAR
# ==============================================================================
MAX_NB, ATOM_FDIM, BOND_FDIM = 10, 31, 6
LIGAND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
AA_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGAND_ELEMENTS) + 
                        onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) + 
                        onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0]) + 
                        onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3]) + 
                        onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other']) + 
                        [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, 
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, 
                        bond.IsInRing(), bond.GetIsConjugated()])

def mol_batch_to_graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds, in_bonds, all_bonds = [], [padding], [], [(-1,-1)]
    lig_scope, total_atoms = [], 0
    for mol in mol_batch:
        if not mol: continue
        n = mol.GetNumAtoms()
        for a in mol.GetAtoms(): fatoms.append(atom_features(a)); in_bonds.append([])
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx()+total_atoms, b.GetEndAtomIdx()+total_atoms
            idx = len(all_bonds); all_bonds.append((u,v)); fbonds.append(torch.cat([fatoms[u], bond_features(b)], 0)); in_bonds[v].append(idx)
            idx = len(all_bonds); all_bonds.append((v,u)); fbonds.append(torch.cat([fatoms[v], bond_features(b)], 0)); in_bonds[u].append(idx)
        lig_scope.append((total_atoms, n)); total_atoms += n
    if not fatoms: return None
    fatoms, fbonds = torch.stack(fatoms, 0), torch.stack(fbonds, 0)
    ag, bg = torch.zeros(total_atoms, MAX_NB).long(), torch.zeros(len(all_bonds), MAX_NB).long()
    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]): 
            if i < MAX_NB: ag[a, i] = b
    for b1 in range(1, len(all_bonds)):
        u, v = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[u]):
            if i < MAX_NB and all_bonds[b2][0] != v: bg[b1, i] = b2
    return (fatoms, fbonds, ag, bg, lig_scope)

# --- PCA LOAD ---
PCA_MEAN, PCA_TRANSFORM = None, None
def load_pca_params():
    global PCA_MEAN, PCA_TRANSFORM
    if os.path.exists(PCA_PARAMS_PATH):
        with open(PCA_PARAMS_PATH, 'rb') as f:
            params = pickle.load(f)
            if isinstance(params, dict):
                PCA_MEAN, PCA_TRANSFORM = params['mean'], params['components']
            else:
                PCA_MEAN, PCA_TRANSFORM = params.mean_, params.components_
        print("✅ PCA Parametreleri yüklendi.", flush=True)

load_pca_params()

def apply_pca(embedding_1280d):
    if PCA_TRANSFORM is None: return embedding_1280d[:512]
    return np.dot(embedding_1280d - PCA_MEAN, PCA_TRANSFORM.T).astype(np.float32)

def move_to_device(data, device):
    if isinstance(data, torch.Tensor): return data.to(device)
    if isinstance(data, (list, tuple)): return [move_to_device(t, device) for t in data]
    return data

# ==============================================================================
# 3. VERİ İŞLEME SINIFLARI
# ==============================================================================
def parse_mapping(pdb_file):
    mapping, idx, seen = {}, 0, set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                res_id_str = line[22:26].strip()
                match = re.search(r'\d+', res_id_str)
                if not match: continue
                res_id, chain = int(match.group()), line[21].strip() or 'A'
                key = (chain, res_id)
                if key not in seen and line[17:20].strip() in AA_3_TO_1:
                    mapping[key] = idx
                    seen.add(key); idx += 1
    return mapping

class PocketProcessor():
    def __init__(self, protein_file, crystal_ligand_path, esm_embedding):
        self.fresidues, self.alpha_coordinates, self.valid = [], [], False
        
        target_pdb = protein_file.replace('.mol2', '.pdb')
        target_sdf = crystal_ligand_path.replace('.mol2', '.sdf')
        
        try:
            prot = Chem.MolFromPDBFile(target_pdb, sanitize=False)
            if os.path.exists(target_sdf):
                suppl = Chem.SDMolSupplier(target_sdf, sanitize=False)
                ref_mol = suppl[0] if len(suppl) > 0 else None
            else:
                ref_mol = Chem.MolFromMol2File(crystal_ligand_path, sanitize=False)

            if prot is None or ref_mol is None: return

            center = np.mean(ref_mol.GetConformer().GetPositions(), axis=0)
            mapping = parse_mapping(target_pdb)
            conf_prot = prot.GetConformer()
            res_dict = defaultdict(list)
            
            for atom in prot.GetAtoms():
                pos = conf_prot.GetAtomPosition(atom.GetIdx())
                if np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - center) <= 12.0:
                    info = atom.GetPDBResidueInfo()
                    if info:
                        res_key = (info.GetChainId().strip() or 'A', info.GetResidueNumber())
                        res_dict[res_key].append([pos.x, pos.y, pos.z])
            
            for res_key, coords in res_dict.items():
                feat = np.zeros(512, dtype=np.float32)
                if esm_embedding is not None and res_key in mapping:
                    idx = mapping[res_key]
                    if idx < len(esm_embedding):
                        feat = apply_pca(esm_embedding[idx])
                
                self.fresidues.append(feat)
                self.alpha_coordinates.append(np.mean(coords, axis=0).reshape(1,3))

            if self.fresidues:
                self.fresidues = torch.from_numpy(np.array(self.fresidues)).float()
                self.alpha_coordinates = torch.from_numpy(np.concatenate(self.alpha_coordinates, axis=0)).float()
                self.valid = True
                
        except Exception as e:
            print(f"      ⚠️ Pocket Hatası ({os.path.basename(protein_file)}): {e}")
            self.valid = False

class LigandDataset(Dataset):
    def __init__(self, sdf_file):
        self.mols = []
        try:
            if sdf_file.endswith('.gz'):
                with gzip.open(sdf_file, 'rb') as f:
                    suppl = Chem.ForwardSDMolSupplier(f, sanitize=False)
                    self.mols = [m for m in suppl if m]
            else:
                suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
                self.mols = [m for m in suppl if m]
        except: pass
    def __len__(self): return len(self.mols)
    def __getitem__(self, idx): return self.mols[idx]

def calculate_ef_metrics(act_scores, dec_scores):
    y_true = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
    y_score = np.concatenate([act_scores, dec_scores])
    sorted_labels = y_true[np.argsort(-y_score)]
    n_total, n_actives = len(y_score), len(act_scores)
    res = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n = math.ceil(n_total * pct)
        res[f'EF_{pct*100}%'] = (sorted_labels[:n].sum() / n) / (n_actives / n_total) if n > 0 and n_actives > 0 else 0.0
    return res

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('-h5', '--h5_path', required=True)
    parser.add_argument('--json_path', type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-s','--feature_dims', type=int, default=512)
    parser.add_argument('--out_name', type=str, default='benchmark_LIT_PCBA_ESM')
    parser.add_argument('--device', type=str, default='')
    
    # --- YENİ EKLENEN (DUMMY) PARAMETRELER (Hata vermemesi için) ---
    parser.add_argument('--gating', action='store_true')
    parser.add_argument('--vector', action='store_true')

    args = parser.parse_args()

    if args.device: device = torch.device(args.device)
    else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu': torch.set_num_threads(os.cpu_count())

    os.makedirs(args.out_dir, exist_ok=True)
    wandb.init(project="PLANET_Eval_ESM_LIT_PCBA", name=f"Run_{args.out_name}", config=args)

    print(f"⚙️ Model Yükleniyor: {args.model_path}")
    
    # Model init
    model = PLANET(args.feature_dims, 8, 128, 128, 3, 10, 1, str(device))
    
    # strict=False
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print("✅ Model ağırlıkları başarıyla yüklendi (strict=False).")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        sys.exit(1)

    model.to(device).eval()

    if not os.path.exists(args.h5_path): sys.exit("❌ H5 Yok")
    h5_file = h5py.File(args.h5_path, 'r')
    
    with open(args.json_path, 'r') as f: data_map = json.load(f)
    
    csv_path = os.path.join(args.out_dir, f'{args.out_name}.csv')
    completed, results = set(), []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path); results = df.to_dict('records')
            completed = set(df['Run_ID'].tolist())
        except: pass

    for target, templates in data_map.items():
        for record in templates:
            run_id = f"{target}_{record['name']}"
            if run_id in completed: continue
            
            pdb_id = record['name']
            target_esm = None
            for k in [pdb_id.lower(), pdb_id.upper()]:
                if k in h5_file:
                    target_esm = h5_file[k][:]
                    break
            
            if target_esm is None:
                print(f"⚠️ Embedding H5 dosyasında bulunamadı: {pdb_id}"); continue

            print(f"\n--- {run_id} İşleniyor (ESM-2) ---", flush=True)
            protein_path = record.get('protein_pdb') or record.get('protein_file')
            pocket = PocketProcessor(protein_path, record['ligand_file'], target_esm)
            
            if not pocket.valid:
                print(f"    ❌ Pocket hatası: {run_id}"); continue
            
            p_fres, p_coords = pocket.fresidues.to(device), pocket.alpha_coordinates.to(device)

            # 🎯 DÜZELTME: p_fres ve p_coords argüman olarak gönderiliyor
            def predict(loader, p_fres, p_coords):
                scores = []
                with torch.no_grad():
                    for batch in loader:
                        if not batch: continue
                        data = [move_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in batch]
                        bs = len(data[-1])
                        # Artık p_fres ve p_coords lokal değişken
                        res_scope = [(i*p_fres.shape[0], p_fres.shape[0]) for i in range(bs)]
                        try:
                            _, _, affs = model((p_fres.repeat(bs, 1), None, res_scope, p_coords.repeat(bs, 1)), data)
                            scores.append(affs.cpu().numpy().flatten())
                        except: pass
                return np.concatenate(scores) if scores else []

            # predict çağırırken p_fres ve p_coords gönderiliyor
            act_scores = predict(DataLoader(LigandDataset(record['actives_file']), batch_size=args.batch_size, collate_fn=mol_batch_to_graph), p_fres, p_coords)
            dec_scores = predict(DataLoader(LigandDataset(record['decoys_file']), batch_size=args.batch_size, collate_fn=mol_batch_to_graph), p_fres, p_coords)

            if len(act_scores) > 0 and len(dec_scores) > 0:
                y_true = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
                y_score = np.concatenate([act_scores, dec_scores])
                roc_auc = roc_auc_score(y_true, y_score)
                ef = calculate_ef_metrics(act_scores, dec_scores)
                
                log_entry = {"Run_ID": run_id, "Target": target, "Template": record['name'], "ROC_AUC": roc_auc}
                log_entry.update(ef); wandb.log(log_entry); results.append(log_entry)
                pd.DataFrame(results).to_csv(csv_path, index=False)
                print(f"    🎯 AUC: {roc_auc:.4f} | EF_1%: {ef['EF_1.0%']:.2f}", flush=True)

    h5_file.close(); wandb.finish()