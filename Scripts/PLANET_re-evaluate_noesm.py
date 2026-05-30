import torch
import os, sys, argparse, re
import numpy as np
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

# --- IMPORTLAR ---
from PLANET_datautils import ProLigDataset
from PLANET_model import PLANET

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        res = [move_to_device(t, device) for t in data]
        return tuple(res) if isinstance(data, tuple) else res
    return data

def get_iteration(filename):
    match = re.search(r'iter-(\d+)', filename)
    return int(match.group(1)) if match else 0

def run_evaluation(model, loader, device, name):
    # Global Listeler (1:11 Dengesiz Set)
    all_p_true_global, all_p_pred_global = [], []
    # Binder-Only Listeler (Sadece Binder Contactları)
    all_p_true_binder, all_p_pred_binder = [], []
    
    all_affinity_true, all_affinity_pred = [], []
    all_pK_flags = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None: continue
            res_b, mol_b, tgts = batch_data
            
            res_b = move_to_device(res_b, device)
            mol_b = move_to_device(mol_b, device)
            tgts  = move_to_device(tgts, device)
            
            preds = model(res_b, mol_b)
            
            p_pred = preds[1].detach().cpu()
            p_true = tgts[1].detach().cpu()
            
            # --- GLOBAL VERİ TOPLAMA ---
            all_p_pred_global.append(p_pred)
            all_p_true_global.append(p_true)
            
            # --- BINDER-ONLY DİLİMLEME (SLICING) ---
            # DataUtils: res_b[2] = res_scope, mol_b içinde bir liste = mol_scope
            res_scope = res_b[2]
            mol_scope = next(x for x in mol_b if isinstance(x, list))
            
            binder_atoms = mol_scope[0][1] # İlk molekül (binder) atom sayısı
            pocket_res   = res_scope[0][1] # Proteindeki residue sayısı
            binder_len   = binder_atoms * pocket_res
            
            all_p_pred_binder.append(p_pred[:binder_len])
            all_p_true_binder.append(p_true[:binder_len])
            
            # Afinite verileri
            all_affinity_pred.append(preds[2].detach().cpu())
            all_affinity_true.append(tgts[2].detach().cpu())
            all_pK_flags.append(tgts[3].detach().cpu())

    def get_metrics(y_true_list, y_pred_list, prefix):
        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()
        
        auroc = roc_auc_score(y_true, y_pred) if np.sum(y_true) > 0 else 0.5
        auprc = average_precision_score(y_true, y_pred)
        y_bin = (y_pred > 0.5).astype(int)
        f1   = f1_score(y_true, y_bin, zero_division=0)
        prec = precision_score(y_true, y_bin, zero_division=0)
        rec  = recall_score(y_true, y_bin, zero_division=0)
        
        return {
            f"{name}/{prefix}_AUROC": auroc,
            f"{name}/{prefix}_AUPRC": auprc,
            f"{name}/{prefix}_F1": f1,
            f"{name}/{prefix}_Precision": prec,
            f"{name}/{prefix}_Recall": rec
        }

    # Metrikleri hesapla
    m_global = get_metrics(all_p_true_global, all_p_pred_global, "Global")
    m_binder = get_metrics(all_p_true_binder, all_p_pred_binder, "BinderOnly")
    
    # Afinite korelasyonu (Sadece Binderlar)
    aff_true = torch.cat(all_affinity_true).numpy()
    aff_pred = torch.cat(all_affinity_pred).numpy()
    pk_f     = torch.cat(all_pK_flags).numpy()
    mask     = pk_f == 1.0
    p_corr   = pearsonr(aff_true[mask], aff_pred[mask])[0] if mask.sum() > 1 else 0.0

    return {**m_global, **m_binder, f"{name}/Affinity_Pearson": p_corr}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--valid_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--project', default="PLANET_EVOLUTION")
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--dims', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=args.project, name=args.run_name)

    model = PLANET(args.dims, 8, 128, 128, 3, 10, 1, device).to(device)

    ckpts = sorted([f for f in os.listdir(args.ckpt_dir) if "iter-" in f], key=get_iteration)

    v_loader = DataLoader(ProLigDataset(args.valid_data, 1, decoy_flag=True, decoy_count=11), 
                          batch_size=1, collate_fn=lambda x:x[0], num_workers=4)
    t_loader = DataLoader(ProLigDataset(args.test_data, 1, decoy_flag=True, decoy_count=11), 
                          batch_size=1, collate_fn=lambda x:x[0], num_workers=4)

    for ckpt in ckpts:
        step = get_iteration(ckpt)
        print(f"⌛ Analiz Başladı: {ckpt}")
        
        # WandB'ye "çalışıyorum" sinyali gönder
        wandb.log({"status/is_evaluating": 1}, step=step, commit=False)
        
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, ckpt), map_location=device))
        
        v_res = run_evaluation(model, v_loader, device, "Valid")
        t_res = run_evaluation(model, t_loader, device, "Test")
        metrics = {**v_res, **t_res}
        
        # Terminal çıktısı (Özet)
        print(f"✅ {ckpt} | Global_AUROC: {metrics['Valid/Global_AUROC']:.3f} | BinderOnly_AUROC: {metrics['Valid/BinderOnly_AUROC']:.3f}")
        
        # WandB'ye zorla gönder (Flush)
        wandb.log(metrics, step=step, commit=True)

    wandb.finish()