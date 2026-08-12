import torch
import torch.nn as nn
import numpy as np
import argparse, os, sys, json, gzip, pandas as pd, math, wandb, random, re
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from collections import defaultdict
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- 1. PROJE YOLU VE MODÜL AYARLARI ---
MODEL_DIR_PATH = os.path.join(PLANET_ROOT, "Model") 
DEFAULT_JSON_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_arf.json")

if MODEL_DIR_PATH not in sys.path: 
    sys.path.append(MODEL_DIR_PATH)

try:
    from PLANET_model import PLANET
    print("✅ Standart PLANET_model YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# 2. CHEMUTILS & YARDIMCI FONKSİYONLAR (BOYUT 20 İÇİN SABİT)
# ==============================================================================
MAX_NB, ATOM_FDIM, BOND_FDIM = 10, 31, 6
LIGAND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
AA_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

def move_to_device(data, device):
    if isinstance(data, torch.Tensor): return data.to(device)
    if isinstance(data, (list, tuple)): return [move_to_device(t, device) for t in data]
    return data

def get_residue_feature(res_name_3, feature_dim=20):
    res_1 = AA_3_TO_1.get(res_name_3, 'X')
    vec = np.zeros(feature_dim, dtype=np.float32)
    if res_1 in AA_LIST: vec[AA_LIST.index(res_1)] = 1.0
    return vec

def atom_features(atom):
    def onek(x, s): return [int(x==i) for i in s] if x in s else [int(x==s[-1]) for i in s]
    return torch.Tensor(onek(atom.GetSymbol(), LIGAND_ELEMENTS) + onek(atom.GetDegree(), [0,1,2,3,4,5]) + onek(atom.GetFormalCharge(), [-1,-2,1,2,0]) + onek(int(atom.GetChiralTag()), [0,1,2,3]) + onek(str(atom.GetHybridization()),['SP','SP2','SP3','other']) + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(), bond.GetIsConjugated()])

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

# ==============================================================================
# 3. VERİ İŞLEME SINIFLARI
# ==============================================================================
class PocketProcessor():
    def __init__(self, protein_file, crystal_ligand_path, feature_dim=20):
        self.fresidues, self.alpha_coordinates, self.valid = [], [], False
        # PDB öncelikli okuma
        pdb_path = protein_file.replace('.mol2', '.pdb')
        sdf_path = crystal_ligand_path.replace('.mol2', '.sdf')
        
        try:
            target_p = pdb_path if os.path.exists(pdb_path) else protein_file
            prot = Chem.MolFromPDBFile(target_p, sanitize=False) if target_p.endswith('.pdb') else Chem.MolFromMol2File(target_p, sanitize=False)
            
            target_l = sdf_path if os.path.exists(sdf_path) else crystal_ligand_path
            if target_l.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(target_l, sanitize=False)
                ref_mol = suppl[0] if len(suppl) > 0 else None
            else:
                ref_mol = Chem.MolFromMol2File(target_l, sanitize=False)

            if prot is None or ref_mol is None: return

            center = np.mean(ref_mol.GetConformer().GetPositions(), axis=0)
            conf_prot = prot.GetConformer()
            res_dict = defaultdict(list)
            for atom in prot.GetAtoms():
                pos = conf_prot.GetAtomPosition(atom.GetIdx())
                if np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - center) <= 12.0:
                    info = atom.GetPDBResidueInfo()
                    if info:
                        name = info.GetResidueName().strip()
                        if name == 'MSE': name = 'MET'
                        res_dict[(info.GetResidueNumber(), info.GetChainId(), name)].append([pos.x, pos.y, pos.z])
            
            for (_, _, name), coords in res_dict.items():
                if name in AA_3_TO_1:
                    self.fresidues.append(get_residue_feature(name, feature_dim))
                    self.alpha_coordinates.append(np.mean(coords, axis=0).reshape(1,3))

            if self.fresidues:
                self.fresidues = torch.from_numpy(np.array(self.fresidues)).float()
                self.alpha_coordinates = torch.from_numpy(np.concatenate(self.alpha_coordinates, axis=0)).float()
                self.valid = True
        except: self.valid = False

class LigandDataset(Dataset):
    def __init__(self, sdf_file):
        self.mols = []
        print(f"  ⏳ {os.path.basename(sdf_file)} okunuyor...", flush=True)
        try:
            if sdf_file.endswith('.gz'):
                with gzip.open(sdf_file, 'rb') as f:
                    suppl = Chem.ForwardSDMolSupplier(f, sanitize=False)
                    for i, m in enumerate(suppl):
                        if m: self.mols.append(m)
                        if i > 0 and i % 10000 == 0: print(f"    ✅ {i} molekül yüklendi...", flush=True)
            else:
                suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
                for i, m in enumerate(suppl):
                    if m: self.mols.append(m)
                    if i > 0 and i % 10000 == 0: print(f"    ✅ {i} molekül yüklendi...", flush=True)
        except Exception as e:
            print(f"      ❌ Okuma Hatası: {e}")
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
    parser.add_argument('--out_name', type=str, default='benchmark_LIT_PCBA')
    parser.add_argument('--json_path', type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    wandb.init(project="PLANET_Eval_LIT_PCBA", name=f"Run_{args.out_name}", config=args)

    model = PLANET(args.feature_dims, 8, 300, 300, 3, 10, 1, str(device))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    with open(args.json_path, 'r') as f: data_map = json.load(f)
    csv_path = os.path.join(args.out_dir, f'{args.out_name}.csv')
    completed, results = set(), []
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path); results = existing.to_dict('records')
        completed = set(existing['Run_ID'].tolist())

    for target, templates in data_map.items():
        for record in templates:
            run_id = f"{target}_{record['name']}"
            if run_id in completed: continue
            
            print(f"\n--- {run_id} İşleniyor ---", flush=True)
            pocket = PocketProcessor(record['protein_file'], record['ligand_file'], 20)
            if not pocket.valid: print(f"    ⚠️ Atlanıyor: {run_id} (Pocket geçersiz)"); continue
            
            p_fres, p_coords = pocket.fresidues.to(device), pocket.alpha_coordinates.to(device)

            def predict(loader):
                scores = []
                with torch.no_grad():
                    for batch in loader:
                        if not batch: continue
                        data = [move_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in batch]
                        bs = len(data[-1])
                        res_scope = [(i*p_fres.shape[0], p_fres.shape[0]) for i in range(bs)]
                        _, _, affs = model((p_fres.repeat(bs, 1), None, res_scope, p_coords.repeat(bs, 1)), data)
                        scores.append(affs.cpu().numpy().flatten())
                return np.concatenate(scores) if scores else []

            act_scores = predict(DataLoader(LigandDataset(record['actives_file']), batch_size=args.batch_size, collate_fn=mol_batch_to_graph))
            dec_scores = predict(DataLoader(LigandDataset(record['decoys_file']), batch_size=args.batch_size, collate_fn=mol_batch_to_graph))

            if len(act_scores) > 0 and len(dec_scores) > 0:
                y_true = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
                y_score = np.concatenate([act_scores, dec_scores])
                roc_auc = roc_auc_score(y_true, y_score)
                ef = calculate_ef_metrics(act_scores, dec_scores)
                log_entry = {"Run_ID": run_id, "Target": target, "Template": record['name'], "ROC_AUC": roc_auc}
                log_entry.update(ef); wandb.log(log_entry); results.append(log_entry)
                pd.DataFrame(results).to_csv(csv_path, index=False)
                print(f"    🎯 AUC: {roc_auc:.4f} | EF_1%: {ef['EF_1.0%']:.2f}", flush=True)

    wandb.finish()