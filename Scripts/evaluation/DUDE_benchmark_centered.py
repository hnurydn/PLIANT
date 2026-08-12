"""
DUDE_benchmark_unified.py
==========================
DUD-E benchmark -- supports all model types.
--model_type: noesm | esm | vg | contrastive
"""

import torch
import numpy as np
import argparse, os, sys, json, gzip, pickle, lmdb, math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import Chem, RDLogger
import wandb

RDLogger.DisableLog('rdApp.*')

# ─── Dinamik model import ──────────────────────────────────────────────────────
def _get_arg(name):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_mt = _get_arg('--model_type') or 'esm'
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

if _mt == 'noesm':
    from PLANET_model import PLANET
    print('✅ PLANET_model (NoESM) yüklendi.')
elif _mt == 'esm':
    from PLANET_model_esm2 import PLANET
    print('✅ PLANET_model_esm2 yüklendi.')
elif _mt == 'vg':
    from PLANET_model_esm2_vector_gating import PLANET
    print('✅ PLANET_model_esm2_vector_gating yüklendi.')
elif _mt == 'contrastive':
    from PLANET_model_esm2_vector_gating_contrastive_loss import PLANET
    print('✅ PLANET_model_esm2_vector_gating_contrastive_loss yüklendi.')
else:
    print(f'❌ Bilinmeyen model_type: {_mt}'); sys.exit(1)

# ─── Pocket sınıfı (pickle deserialize için) ──────────────────────────────────
class DudePocketRecord:
    def __init__(self):
        self.pocket_residues   = []
        self.res_features      = None
        self.alpha_coordinates = None
        self.res_count         = 0
        self.actives_file      = ''
        self.decoys_file       = ''
        self.target_name       = ''

class SimpleResidue:
    def __init__(self, residue_lines=None):
        self.mass_center    = None
        self.coordinates    = None
        self.alpha_position = None
        self.feature        = None
    def get_feature(self):        return self.feature
    def get_alpha_position(self): return self.alpha_position
    def get_mass_center(self):    return self.mass_center

# ─── Ligand graph ──────────────────────────────────────────────────────────────
MAX_NB          = 10
LIGAND_ELEMENTS = ['C','N','O','S','F','P','Cl','Br','I','H','unknown']
ATOM_FDIM       = len(LIGAND_ELEMENTS) + 6 + 5 + 4 + 4 + 1
BOND_FDIM       = 6

def onek_encoding_unk(x, s):
    if x not in s: x = s[-1]
    return [int(x == i) for i in s]

def atom_features(atom):
    return torch.Tensor(
        onek_encoding_unk(atom.GetSymbol(), LIGAND_ELEMENTS)
        + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
        + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
        + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
        + onek_encoding_unk(str(atom.GetHybridization()), ['SP','SP2','SP3','other'])
        + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing(), bond.GetIsConjugated()])

def mol_batch_to_graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds = [], [padding]
    in_bonds, all_bonds = [], [(-1,-1)]
    lig_scope = []
    lig_coordinates = []
    total_atoms = 0
    for mol in mol_batch:
        try: Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        except: pass
        n_atoms = mol.GetNumAtoms()
        try:
            conf = mol.GetConformer()
        except:
            conf = None
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom)); in_bonds.append([])
            if conf is not None:
                pos = conf.GetAtomPosition(atom.GetIdx())
                lig_coordinates.append([pos.x, pos.y, pos.z])
            else:
                lig_coordinates.append([0.0, 0.0, 0.0])
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() + total_atoms
            a2 = bond.GetEndAtomIdx()   + total_atoms
            b = len(all_bonds); all_bonds.append((a1,a2))
            fbonds.append(torch.cat([fatoms[a1], bond_features(bond)],0)); in_bonds[a2].append(b)
            b = len(all_bonds); all_bonds.append((a2,a1))
            fbonds.append(torch.cat([fatoms[a2], bond_features(bond)],0)); in_bonds[a1].append(b)
        lig_scope.append((total_atoms, n_atoms)); total_atoms += n_atoms
    if not fatoms: return None
    fatoms = torch.stack(fatoms,0); fbonds = torch.stack(fbonds,0)
    total_bonds = len(all_bonds)
    agraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()
    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]):
            if i < MAX_NB: agraph[a,i] = b
    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if i < MAX_NB and all_bonds[b2][0] != y: bgraph[b1,i] = b2
    lig_coordinates = torch.tensor(lig_coordinates, dtype=torch.float32)
    return (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates)

def collate_ligands(batch):
    valid = [m for m in batch if m is not None]
    return mol_batch_to_graph(valid) if valid else None

# ─── Ligand dataset ────────────────────────────────────────────────────────────
class LigandDataset(Dataset):
    def __init__(self, sdf_path, max_limit=None):
        self.mols = []
        if not os.path.exists(sdf_path): return
        try:
            if sdf_path.endswith('.gz'):
                with gzip.open(sdf_path, 'rb') as f:
                    suppl = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False)
                    for mol in suppl:
                        if mol: self.mols.append(mol)
            else:
                suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
                for mol in suppl:
                    if mol: self.mols.append(mol)
        except Exception as e:
            print(f'  ⚠️  SDF okuma hatası: {e}')
        if max_limit and len(self.mols) > max_limit:
            import random; random.seed(42)
            self.mols = random.sample(self.mols, max_limit)
    def __len__(self):        return len(self.mols)
    def __getitem__(self, i): return self.mols[i]

# ─── LMDB reader ──────────────────────────────────────────────────────────────
class DudeLmdbReader:
    def __init__(self, lmdb_dir):
        lmdb_path = os.path.join(lmdb_dir, 'dude.lmdb')
        meta_path = os.path.join(lmdb_dir, 'dude_meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.lmdb_path = lmdb_path
        self._env = self._txn = self._pid = None
        print(f'✅ LMDB meta: {len(self.meta)} target')

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                  readahead=True, meminit=False, max_readers=64)
            self._txn = self._env.begin(write=False, buffers=True)
            self._pid = pid

    def load(self, target_name):
        self._get_env()
        raw = self._txn.get(target_name.encode())
        return pickle.loads(bytes(raw)) if raw else None

    def __iter__(self): return iter(self.meta)

    def close(self):
        try:
            if self._txn: self._txn.abort()
            if self._env: self._env.close()
        except Exception: pass

# ─── Metrikler ────────────────────────────────────────────────────────────────
def calculate_ef(active_scores, decoy_scores):
    y_true  = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_score = np.concatenate([active_scores, decoy_scores])
    sorted_labels = y_true[np.argsort(-y_score)]
    n_total, n_actives = len(y_score), len(active_scores)
    ef = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n = math.ceil(n_total * pct)
        ef[f'EF_{pct*100:.1f}%'] = (sorted_labels[:n].sum() / n) / (n_actives / n_total) \
            if n > 0 and n_actives > 0 else 0.0
    return ef

# ─── Tahmin ───────────────────────────────────────────────────────────────────
def predict_ligands(model, loader, p_fresidues, p_alpha_coords, device, center_ligands=False):
    scores, n_res = [], p_fresidues.shape[0]
    # Pocket merkezi: alpha carbon koordinatlarının ortalaması
    pocket_center = p_alpha_coords.mean(dim=0)  # (3,)
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            batch_data = batch
            fatoms, fbonds, agraph, bgraph, lig_scope = batch_data[:5]
            lig_coords = batch_data[5] if len(batch_data) == 6 else torch.zeros(fatoms.shape[0], 3)
            fatoms = fatoms.to(device); fbonds = fbonds.to(device)
            agraph = agraph.to(device); bgraph = bgraph.to(device)
            lig_coords = lig_coords.to(device)
            if center_ligands:
                # Her ligandı kendi merkezinden pocket merkezine taşı
                for start, length in lig_scope:
                    mol_coords = lig_coords[start:start+length]
                    mol_center = mol_coords.mean(dim=0)
                    lig_coords[start:start+length] = mol_coords - mol_center + pocket_center
            bs        = len(lig_scope)
            res_scope = [(i * n_res, n_res) for i in range(bs)]
            try:
                _, _, affinities = model(
                    (p_fresidues.repeat(bs,1), None, res_scope, p_alpha_coords.repeat(bs,1)),
                    (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coords))
                scores.append(affinities.cpu().numpy().flatten())
            except Exception as e:
                print(f'    ⚠️  Inference hatası: {e}')
    return np.concatenate(scores) if scores else np.array([])

# ─── Benchmark döngüsü ────────────────────────────────────────────────────────
def run_benchmark(model, reader, out_dir, run_name, args, device):
    csv_path  = os.path.join(out_dir, f'{run_name}.csv')
    completed = set(); results = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            results = df.to_dict('records')
            completed = set(df['Target'].tolist())
            print(f'  📂 Mevcut CSV: {len(completed)} target tamamlanmış.')
        except Exception: pass

    for record in reader:
        target_name  = record['target_name']
        actives_file = record['actives_file']
        decoys_file  = record['decoys_file']
        if target_name in completed: continue

        pocket = reader.load(target_name)
        if pocket is None:
            print(f'  ⚠️  LMDB\'de yok: {target_name}'); continue

        p_fresidues    = torch.from_numpy(pocket.res_features).float().to(device)
        p_alpha_coords = torch.from_numpy(pocket.alpha_coordinates).float().to(device)

        act_ds = LigandDataset(actives_file)
        dec_ds = LigandDataset(decoys_file)
        if len(act_ds) == 0 or len(dec_ds) == 0:
            print(f'  ⚠️  Ligand yok: {target_name}'); continue

        act_loader = DataLoader(act_ds, batch_size=args.batch_size,
                                collate_fn=collate_ligands, num_workers=0)
        dec_loader = DataLoader(dec_ds, batch_size=args.batch_size,
                                collate_fn=collate_ligands, num_workers=0)

        act_scores = predict_ligands(model, act_loader, p_fresidues, p_alpha_coords, device, args.center_ligands)
        dec_scores = predict_ligands(model, dec_loader, p_fresidues, p_alpha_coords, device, args.center_ligands)

        if len(act_scores) == 0 or len(dec_scores) == 0:
            print(f'  ⚠️  Skor yok: {target_name}'); continue

        y_true  = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
        y_score = np.concatenate([act_scores, dec_scores])
        roc_auc = roc_auc_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc  = auc(rec, prec)
        ef      = calculate_ef(act_scores, dec_scores)

        print(f'  🎯 {target_name:20s} | AUC:{roc_auc:.4f} '
              f'EF_1%:{ef["EF_1.0%"]:.2f} act:{len(act_scores)} dec:{len(dec_scores)}')

        log = {'Target': target_name, 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc,
               'n_actives': len(act_scores), 'n_decoys': len(dec_scores)}
        log.update(ef)
        wandb.log(log); results.append(log)
        pd.DataFrame(results).to_csv(csv_path, index=False)

    if results:
        df_all   = pd.DataFrame(results)
        mean_auc = df_all['ROC_AUC'].mean()
        mean_ef1 = df_all['EF_1.0%'].mean()
        print(f'\n  📊 ÖZET [{run_name}]: Mean AUC={mean_auc:.4f} Mean EF_1%={mean_ef1:.2f} ({len(df_all)} target)')
        wandb.log({'mean_ROC_AUC': mean_auc, 'mean_EF_1%': mean_ef1, 'n_targets': len(df_all)})
    return csv_path

# ─── Model yükle ──────────────────────────────────────────────────────────────
def load_model(ckpt_path, args, device):
    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters,
                   args.pro_lig_update_iters, str(device))
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        print(f'  ✅ Yüklendi: {os.path.basename(ckpt_path)}')
    except Exception as e:
        print(f'  ❌ Yükleme hatası: {e}'); return None
    model.to(device).eval()
    return model

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DUDE benchmark — unified')
    parser.add_argument('--model_type', required=True,
                        choices=['noesm','esm','vg','contrastive'])
    parser.add_argument('--lmdb_dir',   required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--feature_dims',         type=int, default=512)
    parser.add_argument('--nheads',               type=int, default=8)
    parser.add_argument('--key_dims',             type=int, default=128)
    parser.add_argument('--value_dims',           type=int, default=128)
    parser.add_argument('--pro_update_inters',    type=int, default=3)
    parser.add_argument('--lig_update_iters',     type=int, default=10)
    parser.add_argument('--pro_lig_update_iters', type=int, default=1)
    parser.add_argument('--batch_size',           type=int, default=64)
    parser.add_argument('--device',               type=str, default='')
    parser.add_argument('--center_ligands', action='store_true',
                        help='Ligand koordinatlarını pocket merkezine taşı (pose-free modu)')
    parser.add_argument('--wandb_project',        type=str, default='PLANET_DUDE')
    parser.add_argument('--run_name',             type=str, default='')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'⚙️  Device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)
    reader = DudeLmdbReader(args.lmdb_dir)

    run_name = args.run_name or f'{args.model_type}_{os.path.basename(args.model_path)}'
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args), reinit=True)

    model = load_model(args.model_path, args, device)
    if model:
        run_benchmark(model, reader, args.out_dir, run_name, args, device)
    wandb.finish()
    reader.close()
    print('\n🎉 Tamamlandı!')