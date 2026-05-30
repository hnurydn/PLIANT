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
import traceback

# --- PATH AYARLARI ---
CHEMUTILS_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation"
MODEL_DIR_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model" 
JSON_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/dude_paths_prepared_arf.json"

if CHEMUTILS_PATH not in sys.path: sys.path.append(CHEMUTILS_PATH)
if MODEL_DIR_PATH not in sys.path: sys.path.append(MODEL_DIR_PATH)

# --- IMPORT KONTROLÜ ---
try:
    from chemutils_dude import ComplexPocket, tensorize_all
    print("✅ chemutils_dude YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: chemutils_dude bulunamadı!", flush=True)
    sys.exit(1)

try:
    from PLANET_model import PLANET
    print("✅ PLANET_model YÜKLENDİ.", flush=True)
except ImportError:
    print("❌ HATA: PLANET_model.py bulunamadı!", flush=True)
    sys.exit(1)

RDLogger.DisableLog('rdApp.*')

# --- DATASET ---
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

class RawSDFDataset(Dataset):
    def __init__(self, sdf_path, protein_path, is_active=False):
        self.protein_path = protein_path
        self.is_active = is_active
        print(f"  ⏳ Okunuyor: {os.path.basename(sdf_path)}...", flush=True)
        self.mols = list(load_sdf_gz_iterator(sdf_path))
        print(f"  ✅ {len(self.mols)} molekül yüklendi.", flush=True)

    def __len__(self): return len(self.mols)

    def __getitem__(self, idx):
        try:
            mol = self.mols[idx]
            pk_val = 5.0 if self.is_active else 0.0
            pocket = ComplexPocket(self.protein_path, mol, pK=pk_val)
            # Kritik: Residue yoksa veya Atom yoksa None dön
            if not pocket.pocket_residues or pocket.ligand.atom_count == 0: return None
            return pocket
        except: return None

# --- COLLATE FUNCTION (JUPYTER ILE AYNI) ---
skipped_count = 0
def safe_collate_fn(batch_list):
    global skipped_count
    valid_batch = []
    for item in batch_list:
        if item is None:
            skipped_count += 1
            continue
        try:
            # Residue kontrolü
            valid_res = 0
            for res in item.pocket_residues:
                pos = res.get_alpha_position()
                if pos is not None and len(pos) > 0:
                    valid_res += 1
            
            if valid_res > 0:
                valid_batch.append(item)
            else:
                skipped_count += 1
        except:
            skipped_count += 1
            continue

    if len(valid_batch) == 0: return None
    
    try:
        return tensorize_all(valid_batch, decoy_flag=False)
    except:
        skipped_count += len(valid_batch)
        return None

# --- INFERENCE ---
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
                
                scores = predicted_affinities.cpu().numpy().flatten()
                # NaN varsa 0.0 yap
                scores = np.nan_to_num(scores, nan=0.0)
                
                all_scores.append(scores)
                
            except Exception as e:
                # print(f"Batch Hatası: {e}")
                continue

    if not all_scores: return np.array([])
    return np.concatenate(all_scores)

# --- METRICS ---
def calculate_metrics(active_scores, decoy_scores):
    if len(active_scores) == 0 or len(decoy_scores) == 0: return None
    y_true = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_scores = np.concatenate([active_scores, decoy_scores])
    
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
    except: return None
    
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

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)

    args = parser.parse_args()
    
    wandb.init(project="PLANET_Benchmark", name=f"Eval_{os.path.basename(args.model_path)}", config=args)
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

    csv_path = os.path.join(args.out_dir, 'benchmark_results.csv')
    if os.path.exists(csv_path): os.remove(csv_path)

    for target_name, records in data_map.items():
        if not records: continue
        record = records[0] 
        print(f"\n--- İşleniyor: {target_name} ---", flush=True)
        
        skipped_count = 0
        try:
            active_ds = RawSDFDataset(record['actives_file'], record['protein_pdb'], is_active=True)
            decoy_ds = RawSDFDataset(record['decoys_file'], record['protein_pdb'], is_active=False)
            
            if len(active_ds) == 0 or len(decoy_ds) == 0: continue

            # NUM_WORKERS=0 (JUPYTER GİBİ GÜVENLİ)
            active_loader = DataLoader(active_ds, batch_size=args.batch_size, shuffle=False, 
                                       collate_fn=safe_collate_fn, num_workers=0)
            decoy_loader = DataLoader(decoy_ds, batch_size=args.batch_size, shuffle=False, 
                                      collate_fn=safe_collate_fn, num_workers=0)
            
            print(f"Actives ({len(active_ds)}) tahmin ediliyor...", flush=True)
            active_scores = run_inference(model, active_loader, device)
            
            print(f"Decoys ({len(decoy_ds)}) tahmin ediliyor...", flush=True)
            decoy_scores = run_inference(model, decoy_loader, device)
            
            print(f"⚠️ Atlanan: {skipped_count} | Actives Skor: {len(active_scores)} | Decoys Skor: {len(decoy_scores)}", flush=True)

            metrics = calculate_metrics(active_scores, decoy_scores)
            
            if metrics:
                roc_auc, pr_auc, ef_dict = metrics
                print(f"🎯 {target_name} -> ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}", flush=True)
                
                log_data = {"Target": target_name, "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "Skipped": skipped_count}
                log_data.update(ef_dict)
                wandb.log(log_data)
                results.append(log_data)
                
                # KAYIT
                df = pd.DataFrame(results)
                df.to_csv(csv_path, index=False)
                print(f"💾 KAYDEDİLDİ: {csv_path}", flush=True)
            else:
                print("⚠️ Metrik hesaplanamadı (Skorlar boş olabilir).", flush=True)
                
        except Exception as e:
            print(f"❌ HATA ({target_name}): {e}", flush=True)

    print(f"\n✅ BİTTİ. Sonuçlar: {csv_path}")
    wandb.finish()
