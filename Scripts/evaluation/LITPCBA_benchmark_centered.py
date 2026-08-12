"""
LITPCBA_benchmark_lmdb.py
==========================
LIT-PCBA benchmark — NoESM ve ESM versiyonları, LMDB tabanlı.

--model_type noesm  → PLANET_model      (feature_dim=20,  BLOSUM62)
--model_type esm    → PLANET_model_esm2 (feature_dim=512, ESM2+PCA)

LIT-PCBA özellikleri:
  - Her target için birden fazla template → run_id = {target}_{template}
  - actives/decoys: düz .sdf (gz değil)
  - Sonuçlar hem run_id hem target bazında raporlanır

Kullanım:
    # NoESM, tek checkpoint:
    python LITPCBA_benchmark_lmdb.py \
        --lmdb_dir   /arf/.../lmdb_litpcba_noesm \
        --model_path /arf/.../model.pt \
        --out_dir    /arf/.../litpcba_results \
        --model_type noesm \
        --feature_dims 300 --nheads 8 --key_dims 300 --value_dims 300

    # ESM, tüm klasör:
    python LITPCBA_benchmark_lmdb.py \
        --lmdb_dir       /arf/.../lmdb_litpcba_esm \
        --checkpoint_dir /arf/.../checkpoints_esm \
        --out_dir        /arf/.../litpcba_results_esm \
        --model_type esm \
        --feature_dims 512 --nheads 8 --key_dims 128 --value_dims 128
"""

import torch
import numpy as np
import argparse, os, sys, json, gzip, pickle, lmdb, math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import Chem, RDLogger
import wandb
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



RDLogger.DisableLog('rdApp.*')

# ─── Pocket sınıfı (pickle deserialize için) ──────────────────────────────────
class LitPcbaPocketRecord:
    def __init__(self):
        self.res_features      = None
        self.alpha_coordinates = None
        self.res_count         = 0
        self.actives_file      = ''
        self.decoys_file       = ''
        self.target_name       = ''
        self.template_name     = ''
        self.run_id            = ''



# ──────────────────────────────────────────────
# MODEL IMPORT  (model_type'a göre seçilir)
# ──────────────────────────────────────────────
MODEL_DIR_PATH = os.path.join(PLANET_ROOT, "Model")
if MODEL_DIR_PATH not in sys.path:
    sys.path.append(MODEL_DIR_PATH)

PLANET = None  # main'de yüklenir


def import_model(model_type):
    global PLANET
    if model_type == 'noesm':
        from PLANET_model import PLANET as P
    elif model_type == 'esm':
        from PLANET_model_esm2 import PLANET as P
    elif model_type == 'vg':
        from PLANET_model_esm2_vector_gating import PLANET as P
    elif model_type == 'contrastive':
        from PLANET_model_esm2_vector_gating_contrastive_loss import PLANET as P
    else:
        print(f'❌ Bilinmeyen model_type: {model_type}'); sys.exit(1)
    PLANET = P
    print(f'✅ {model_type} modeli yüklendi.')


# ──────────────────────────────────────────────
# LİGAND GRAPH
# ──────────────────────────────────────────────
MAX_NB         = 10
LIGAND_ELEMENTS = ['C','N','O','S','F','P','Cl','Br','I','H','unknown']
ATOM_FDIM      = len(LIGAND_ELEMENTS) + 6 + 5 + 4 + 4 + 1
BOND_FDIM      = 6

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
    fatoms, fbonds   = [], [padding]
    in_bonds, all_bonds = [], [(-1,-1)]
    lig_scope = []
    lig_coordinates = []
    total_atoms = 0
    for mol in mol_batch:
        if mol is None: continue
        try: Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        except: pass
        n = mol.GetNumAtoms()
        try:
            conf = mol.GetConformer()
        except:
            conf = None
        for a in mol.GetAtoms():
            fatoms.append(atom_features(a)); in_bonds.append([])
            if conf is not None:
                pos = conf.GetAtomPosition(a.GetIdx())
                lig_coordinates.append([pos.x, pos.y, pos.z])
            else:
                lig_coordinates.append([0.0, 0.0, 0.0])
        for b in mol.GetBonds():
            u = b.GetBeginAtomIdx() + total_atoms
            v = b.GetEndAtomIdx()   + total_atoms
            idx = len(all_bonds); all_bonds.append((u,v))
            fbonds.append(torch.cat([fatoms[u], bond_features(b)], 0)); in_bonds[v].append(idx)
            idx = len(all_bonds); all_bonds.append((v,u))
            fbonds.append(torch.cat([fatoms[v], bond_features(b)], 0)); in_bonds[u].append(idx)
        lig_scope.append((total_atoms, n)); total_atoms += n
    if not fatoms: return None
    fatoms = torch.stack(fatoms, 0); fbonds = torch.stack(fbonds, 0)
    nb = len(all_bonds)
    ag = torch.zeros(total_atoms, MAX_NB).long()
    bg = torch.zeros(nb, MAX_NB).long()
    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]):
            if i < MAX_NB: ag[a, i] = b
    for b1 in range(1, nb):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if i < MAX_NB and all_bonds[b2][0] != y: bg[b1, i] = b2
    lig_coordinates = torch.tensor(lig_coordinates, dtype=torch.float32)
    return (fatoms, fbonds, ag, bg, lig_scope, lig_coordinates)

def collate_ligands(batch):
    valid = [m for m in batch if m is not None]
    return mol_batch_to_graph(valid) if valid else None


# ──────────────────────────────────────────────
# LİGAND DATASET  (.sdf veya .sdf.gz)
# ──────────────────────────────────────────────
class LigandDataset(Dataset):
    def __init__(self, sdf_path):
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
            print(f'  ⚠️  SDF okuma hatası [{os.path.basename(sdf_path)}]: {e}')
    def __len__(self):        return len(self.mols)
    def __getitem__(self, i): return self.mols[i]


# ──────────────────────────────────────────────
# LMDB READER
# ──────────────────────────────────────────────
class LitPcbaLmdbReader:
    def __init__(self, lmdb_dir):
        lmdb_path = os.path.join(lmdb_dir, 'litpcba.lmdb')
        meta_path = os.path.join(lmdb_dir, 'litpcba_meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        self.lmdb_path = lmdb_path
        self._env = self._txn = self._pid = None
        print(f'✅ LMDB meta: {len(self.meta)} (target, template) kaydı')

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                  readahead=True, meminit=False, max_readers=64)
            self._txn = self._env.begin(write=False, buffers=True)
            self._pid = pid

    def load(self, run_id):
        self._get_env()
        raw = self._txn.get(run_id.encode())
        return pickle.loads(bytes(raw)) if raw else None

    def __iter__(self):
        return iter(self.meta)

    def close(self):
        try:
            if self._txn: self._txn.abort()
            if self._env: self._env.close()
        except Exception: pass


# ──────────────────────────────────────────────
# METRİKLER
# ──────────────────────────────────────────────
def calculate_ef_metrics(active_scores, decoy_scores):
    y_true  = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_score = np.concatenate([active_scores, decoy_scores])
    sl      = y_true[np.argsort(-y_score)]
    n_tot, n_act = len(y_score), len(active_scores)
    ef = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n = math.ceil(n_tot * pct)
        ef[f'EF_{pct*100:.1f}%'] = (sl[:n].sum() / n) / (n_act / n_tot) \
            if n > 0 and n_act > 0 else 0.0
    return ef


# ──────────────────────────────────────────────
# TAHMİN
# ──────────────────────────────────────────────
def predict_ligands(model, loader, p_fresidues, p_alpha_coords, device, center_ligands=False):
    scores, n_res = [], p_fresidues.shape[0]
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
                for start, length in lig_scope:
                    mol_coords = lig_coords[start:start+length]
                    mol_center = mol_coords.mean(dim=0)
                    lig_coords[start:start+length] = mol_coords - mol_center + pocket_center
            bs        = len(lig_scope)
            res_scope = [(i * n_res, n_res) for i in range(bs)]
            try:
                _, _, affs = model(
                    (p_fresidues.repeat(bs, 1), None, res_scope,
                     p_alpha_coords.repeat(bs, 1)),
                    (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coords))
                scores.append(affs.cpu().numpy().flatten())
            except Exception as e:
                print(f'    ⚠️  Inference hatası: {e}'); continue
    return np.concatenate(scores) if scores else np.array([])


# ──────────────────────────────────────────────
# BENCHMARK DÖNGÜSÜ
# ──────────────────────────────────────────────
def run_benchmark(model, reader, out_dir, run_name, args, device):
    csv_path  = os.path.join(out_dir, f'{run_name}.csv')
    completed = set(); results = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            results   = df.to_dict('records')
            completed = set(df['Run_ID'].tolist())
            print(f'  📂 Mevcut CSV: {len(completed)} kayıt tamamlanmış.')
        except Exception: pass

    for record in reader:
        run_id       = record['run_id']
        target_name  = record['target_name']
        template_name = record['template_name']
        actives_file = record['actives_file']
        decoys_file  = record['decoys_file']

        if run_id in completed: continue

        pocket = reader.load(run_id)
        if pocket is None:
            print(f'  ⚠️  LMDB\'de yok: {run_id}'); continue

        p_fresidues    = torch.from_numpy(pocket.res_features).float().to(device)
        p_alpha_coords = torch.from_numpy(pocket.alpha_coordinates).float().to(device)

        act_ds = LigandDataset(actives_file)
        dec_ds = LigandDataset(decoys_file)
        if len(act_ds) == 0 or len(dec_ds) == 0:
            print(f'  ⚠️  Ligand yok: {run_id}'); continue

        act_loader = DataLoader(act_ds, batch_size=args.batch_size,
                                collate_fn=collate_ligands, num_workers=0)
        dec_loader = DataLoader(dec_ds, batch_size=args.batch_size,
                                collate_fn=collate_ligands, num_workers=0)

        act_scores = predict_ligands(model, act_loader, p_fresidues, p_alpha_coords, device, args.center_ligands)
        dec_scores = predict_ligands(model, dec_loader, p_fresidues, p_alpha_coords, device, args.center_ligands)

        if len(act_scores) == 0 or len(dec_scores) == 0:
            print(f'  ⚠️  Skor yok: {run_id}'); continue

        y_true  = np.concatenate([np.ones_like(act_scores), np.zeros_like(dec_scores)])
        y_score = np.concatenate([act_scores, dec_scores])
        roc_auc = roc_auc_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc  = auc(rec, prec)
        ef      = calculate_ef_metrics(act_scores, dec_scores)

        print(f'  🎯 {run_id:30s} | AUC:{roc_auc:.4f} '
              f'EF_1%:{ef["EF_1.0%"]:.2f} '
              f'act:{len(act_scores)} dec:{len(dec_scores)}')

        log = {'Run_ID': run_id, 'Target': target_name,
               'Template': template_name,
               'ROC_AUC': roc_auc, 'PR_AUC': pr_auc,
               'n_actives': len(act_scores), 'n_decoys': len(dec_scores)}
        log.update(ef)
        wandb.log(log); results.append(log)
        pd.DataFrame(results).to_csv(csv_path, index=False)

    # Özet: target bazında ortalama AUC
    if results:
        df_all   = pd.DataFrame(results)
        mean_auc = df_all['ROC_AUC'].mean()
        mean_ef1 = df_all['EF_1.0%'].mean()

        # Target bazında grupla (birden fazla template varsa mean al)
        target_auc = df_all.groupby('Target')['ROC_AUC'].mean()
        mean_target_auc = target_auc.mean()

        print(f'\n  📊 ÖZET [{run_name}]:')
        print(f'     Template-level  Mean AUC = {mean_auc:.4f}  '
              f'Mean EF_1% = {mean_ef1:.2f}  ({len(df_all)} kayıt)')
        print(f'     Target-level    Mean AUC = {mean_target_auc:.4f}  '
              f'({len(target_auc)} target)')
        wandb.log({'mean_ROC_AUC_template': mean_auc,
                   'mean_ROC_AUC_target'  : mean_target_auc,
                   'mean_EF_1%'           : mean_ef1,
                   'n_templates'          : len(df_all),
                   'n_targets'            : len(target_auc)})
    return csv_path


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def load_model(ckpt_path, args, device):
    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters,
                   args.pro_lig_update_iters, str(device))
    try:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device), strict=False)
        print(f'  ✅ Yüklendi: {os.path.basename(ckpt_path)}')
    except Exception as e:
        print(f'  ❌ Yükleme hatası: {e}'); return None
    model.to(device).eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LIT-PCBA benchmark — NoESM/ESM, LMDB tabanlı')
    parser.add_argument('--lmdb_dir',    required=True)
    parser.add_argument('--model_type',  required=True, choices=['noesm', 'esm', 'vg', 'contrastive'])
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--model_path',     type=str)
    grp.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--out_dir',     required=True)
    # Model hiperparametreleri
    parser.add_argument('--feature_dims',         type=int, default=512)
    parser.add_argument('--nheads',               type=int, default=8)
    parser.add_argument('--key_dims',             type=int, default=128)
    parser.add_argument('--value_dims',           type=int, default=128)
    parser.add_argument('--pro_update_inters',    type=int, default=3)
    parser.add_argument('--lig_update_iters',     type=int, default=10)
    parser.add_argument('--pro_lig_update_iters', type=int, default=1)
    parser.add_argument('--batch_size',           type=int, default=128)
    parser.add_argument('--device',               type=str, default='')
    parser.add_argument('--center_ligands', action='store_true',
                        help='Ligand koordinatlarını pocket merkezine taşı (pose-free modu)')
    parser.add_argument('--wandb_project',        type=str, default='PLANET_LITPCBA')
    parser.add_argument('--wandb_name',           type=str, default='')
    args = parser.parse_args()

    # Model import
    import_model(args.model_type)

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
    print(f'⚙️  Device: {device}  |  Model: {args.model_type.upper()}')

    os.makedirs(args.out_dir, exist_ok=True)
    reader = LitPcbaLmdbReader(args.lmdb_dir)

    checkpoints = [args.model_path] if args.model_path else \
                  sorted([os.path.join(args.checkpoint_dir, f)
                          for f in os.listdir(args.checkpoint_dir)
                          if f.endswith('.pt')])
    print(f'📁 {len(checkpoints)} checkpoint')

    for ckpt in checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
        wandb_name = args.wandb_name if args.wandb_name else f'{ckpt_name}_{args.model_type}'
        print(f'\n{"="*60}\n  Checkpoint: {ckpt_name}\n{"="*60}')
        wandb.init(project=args.wandb_project, name=wandb_name,
                   config=vars(args), reinit=True)
        model = load_model(ckpt, args, device)
        if model:
            run_benchmark(model, reader, args.out_dir,
                          f'{ckpt_name}_{args.model_type}', args, device)
        wandb.finish()

    reader.close()
    print('\n🎉 Tüm benchmarklar tamamlandı!')