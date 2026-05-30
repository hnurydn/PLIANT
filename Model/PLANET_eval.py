import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse, os, sys, pickle
import numpy as np
import pandas as pd
from rdkit import RDLogger
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error

# --- MODÜL YOLLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
datautils_path = os.path.join(current_dir, '..', 'Dataset_preparation')
if datautils_path not in sys.path:
    sys.path.append(datautils_path)

try:
    from PLANET_datautils import ProLigDataset 
    from PLANET_model import PLANET 
    from chemutils import ComplexPocket, tensorize_all
    RDLogger.DisableLog('rdApp.*') 
except ImportError as e:
    print(f"KRİTİK HATA: {e}")
    sys.exit(1)

def move_to_device(data, device):
    if isinstance(data, torch.Tensor): return data.to(device)
    if isinstance(data, (list, tuple)): return [move_to_device(t, device) for t in data]
    return data

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    return batch[0] if len(batch) > 0 else None

def evaluate_set(data_loader, data_set_name, total_step, PLANET_model, device, wandb_log=False):
    all_affinity_true, all_affinity_pred = [], []
    total_mae, batch_count = 0.0, 0
    
    print(f"\n>>> Başlıyor: {data_set_name}")

    for batch in data_loader:
        if batch is None: continue
        res_batch, mol_batch, targets = batch 
        try:
            res_batch = move_to_device(res_batch, device)
            mol_batch = move_to_device(mol_batch, device)
            targets = move_to_device(targets, device)
            
            predictions = PLANET_model(res_batch, mol_batch) 
            pred_aff = predictions[2].detach().cpu().numpy().flatten()
            true_aff = targets[2].detach().cpu().numpy().flatten()

            all_affinity_pred.extend(pred_aff)
            all_affinity_true.extend(true_aff)
            total_mae += np.abs(pred_aff - true_aff).mean()
            batch_count += 1
        except Exception: continue 

    if not all_affinity_true:
        print(f"[{data_set_name}] Veri okunamadı.")
        return 

    y_true, y_pred = np.array(all_affinity_true), np.array(all_affinity_pred)
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    is_binder = (y_true > 0)
    auroc = roc_auc_score(is_binder, y_pred) if len(np.unique(is_binder)) > 1 else np.nan

    print(f"=== {data_set_name} SONUÇLARI ===")
    print(f"MAE: {total_mae/batch_count:.3f} | RMSE: {rmse:.3f} | Pearson_R: {pearson_r:.3f} | AUC: {auroc:.3f}")

    if wandb_log:
        wandb.log({f"{data_set_name}/Pearson_R": pearson_r, f"{data_set_name}/AUC": auroc if not np.isnan(auroc) else 0}, step=total_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # .sh dosyasındaki argümanları tanıyacak şekilde güncellendi
    parser.add_argument('-v', '--valid', required=True)
    parser.add_argument('-te', '--test', required=True)
    parser.add_argument('-t', '--train', help="Gerekli değil ama hata vermemesi için eklendi")
    parser.add_argument('-d', '--save_dir', required=True)
    parser.add_argument('--load_checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Model Parametreleri
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)

    args = parser.parse_args()
    device = torch.device('cpu') # Loglardan CPU kullandığını anladım

    wandb.init(project="PLANET_Eval_Test", name=f"eval_{os.path.basename(args.load_checkpoint)}", config=args)

    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device).to(device)
    model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    model.eval()

    # 1. TEST: DECOY'LAR VAR (Sanal Tarama AUC için)
    loader_vs = DataLoader(ProLigDataset(args.valid, args.batch_size, decoy_flag=True), 
                           batch_size=1, collate_fn=safe_collate)

    # 2. TEST: DECOY'LAR YOK (Saf Afinite Pearson R için)
    loader_af = DataLoader(ProLigDataset(args.valid, args.batch_size, decoy_flag=False), 
                           batch_size=1, collate_fn=safe_collate)

    with torch.no_grad():
        step = int(args.load_checkpoint.split('-')[-1]) if '-' in args.load_checkpoint else 0
        evaluate_set(loader_vs, "Valid_VS_WithDecoys", step, model, device, True)
        evaluate_set(loader_af, "Valid_AF_OnlyActives", step, model, device, True)

    wandb.finish()