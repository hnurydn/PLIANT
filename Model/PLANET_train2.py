import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import argparse, os, sys, pickle
import numpy as np
import pandas as pd
from rdkit import RDLogger
import wandb
# 🎯 YENİ IMPORT: precision_recall_curve eklendi
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             precision_score, recall_score, mean_squared_error, 
                             precision_recall_curve)

# --- PYTHON PATH GÜNCELLEMESİ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
datautils_path = os.path.join(current_dir, '..', 'Dataset_preparation')
if datautils_path not in sys.path:
    sys.path.append(datautils_path)

# --- KRİTİK IMPORT'LAR ---
try:
    from PLANET_datautils import ProLigDataset 
    from PLANET_model import PLANET 
    from chemutils import ComplexPocket,tensorize_all,role_of_5
    RDLogger.DisableLog('rdApp.*') 
except ImportError as e:
    print(f"KRİTİK HATA: Modül bulunamadı. Detay: {e}", file=sys.stderr)
    sys.exit(1)

def move_to_device(data, device):
    if isinstance(data, torch.Tensor): return data.to(device)
    if isinstance(data, (list, tuple)): return [move_to_device(t, device) for t in data]
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train', required=True)
    parser.add_argument('-v','--valid',required=True)
    parser.add_argument('-te','--test',required=True)
    parser.add_argument('-d','--save_dir', required=True)
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size',type=int,default=24)
    parser.add_argument('--anneal_iter', type=int, default=20000)
    parser.add_argument('--initial_step',type=int,default=0)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--print_iter', type=int, default=1000)
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--fine_tune_affinity', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True) 

    wandb.init(project="PLANET_Training", name=f"run_gpu_batch{args.batch_size}_ep{args.epoch}", config=args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim Cihazı: {device}")
    
    # Veri Yükleyiciler
    train_loader = DataLoader(ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2),
                              batch_size=1, num_workers=16, pin_memory=True, collate_fn=lambda x:x[0])
    valid_loader = DataLoader(ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True),
                              batch_size=1, num_workers=4, collate_fn=lambda x:x[0])
    test_loader = DataLoader(ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True),
                             batch_size=1, num_workers=4, collate_fn=lambda x:x[0])

    # Model Başlatma
    PLANET_model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                          args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device).to(device)
    
    for param in PLANET_model.parameters():
        if param.dim() == 1: nn.init.constant_(param, 0)
        elif param.dim() >= 2: nn.init.xavier_uniform_(param)
            
    print ("Model #Params: {:d}K".format(sum([x.nelement() for x in PLANET_model.parameters()]) // 1000))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET_model.parameters()), lr=args.lr)

    if args.load_checkpoint:
        PLANET_model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    
    total_step, meters, beta = args.initial_step, np.zeros(6), 0
    PLANET_model.train()

    for epoch in range(1, 1 + args.epoch):
        for batch_data in train_loader:
            try:
                if batch_data is None: continue
                res_f, mol_f, targets = move_to_device(batch_data, device)
                
                optimizer.zero_grad()
                predictions = PLANET_model(res_f, mol_f)
                lig_l, pl_l, aff_l = PLANET_model.compute_loss(predictions, targets, res_f, mol_f)

                # Uncertainty Weighting Loss
                beta = min(1.0, max(0.0, (total_step - 500) / (args.anneal_iter - 500))) if total_step > 500 else 0
                total_loss = (torch.exp(-PLANET_model.log_sigma_lig) * lig_l + 
                              torch.exp(-PLANET_model.log_sigma_pl) * pl_l + 
                              torch.exp(-PLANET_model.log_sigma_affinity) * beta * aff_l + 
                              PLANET_model.log_sigma_lig + PLANET_model.log_sigma_pl + PLANET_model.log_sigma_affinity)
                
                total_loss.backward()
                nn.utils.clip_grad_norm_(PLANET_model.parameters(), args.clip_norm)
                optimizer.step()
                
                # Metrik Güncelleme
                lig_acc, pl_acc, aff_mae = PLANET_model.compute_metrics(predictions, targets)
                meters += np.array([lig_l.item(), lig_acc.item(), pl_l.item(), pl_acc.item(), aff_l.item(), aff_mae.item()])
                total_step += 1

                if total_step % args.print_iter == 0:
                    meters /= args.print_iter
                    print(f"[{total_step}] MAE:{meters[5]:.3f} Beta:{beta:.2f} Sigma_Aff:{torch.exp(PLANET_model.log_sigma_affinity).item():.4f}")
                    wandb.log({"Train/MAE": meters[5], "Hyperparams/Beta": beta}, step=total_step)
                    meters *= 0

                # --- VALIDASYON VE TEST KONTROLÜ ---
                if total_step > 0 and total_step % args.save_iter == 0:
                    PLANET_model.eval()
                    torch.save(PLANET_model.state_dict(), args.save_dir + "/PLANET.iter-" + str(total_step))
                    
                    def evaluate_set(data_loader, data_set_name, total_step, wandb_log=False):
                        all_prolig_true, all_prolig_pred = [], []
                        all_affinity_true, all_affinity_pred = [], []
                        current_meters = np.zeros(6)
                        batch_count = 0 

                        for (res_b, mol_b, targets_b) in data_loader:
                            try:
                                res_b, mol_b, targets_b = move_to_device(res_b, device), move_to_device(mol_b, device), move_to_device(targets_b, device)
                                preds_b = PLANET_model(res_b, mol_b) 
                                l_l, p_l, a_l = PLANET_model.compute_loss(preds_b, targets_b, res_b, mol_b)
                                a_lig, a_pl, m_aff = PLANET_model.compute_metrics(preds_b, targets_b)
                                
                                all_prolig_pred.append(preds_b[1].detach())
                                all_prolig_true.append(targets_b[1].detach())
                                all_affinity_pred.append(preds_b[2].detach())
                                all_affinity_true.append(targets_b[2].detach())

                                current_meters += np.array([l_l.item(), a_lig.item(), p_l.item(), a_pl.item(), a_l.item(), m_aff.item()])
                                batch_count += 1
                            except: continue

                        if batch_count == 0: return

                        # TENSÖRLERİ BİRLEŞTİR
                        prolig_true_flat = torch.cat(all_prolig_true).cpu().numpy()
                        prolig_pred_flat = torch.cat(all_prolig_pred).cpu().numpy()
                        affinity_true_flat = torch.cat(all_affinity_true).cpu().numpy()
                        affinity_pred_flat = torch.cat(all_affinity_pred).cpu().numpy()
                        current_meters /= batch_count

                        # 🎯 ANALİZ: 0.5 EŞİĞİ VE BEST THRESHOLD
                        THRESHOLD = 0.5 
                        prolig_pred_binary = (prolig_pred_flat > THRESHOLD).astype(int)
                        pearson_r = np.corrcoef(affinity_true_flat.flatten(), affinity_pred_flat.flatten())[0, 1] if np.std(affinity_true_flat) > 0 else 0
                        
                        if np.sum(prolig_true_flat) > 0:
                            auroc = roc_auc_score(prolig_true_flat, prolig_pred_flat)
                            auprc = average_precision_score(prolig_true_flat, prolig_pred_flat)
                            f1 = f1_score(prolig_true_flat, prolig_pred_binary, zero_division=0)
                            
                            # --- 🎯 YENİ: DINAMIK EŞİK HESABI ---
                            prec_v, rec_v, thrs_v = precision_recall_curve(prolig_true_flat, prolig_pred_flat)
                            f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-8)
                            best_idx = np.argmax(f1_v)
                            max_f1 = f1_v[best_idx]
                            best_thr = thrs_v[best_idx] if best_idx < len(thrs_v) else 0.5
                            # -----------------------------------
                        else:
                            auroc, auprc, f1, max_f1, best_thr = 0.5, 0.0, 0.0, 0.0, 0.5

                        # SONUÇLARI YAZDIR
                        print (f"[{data_set_name}_{total_step}]\tMAE:{current_meters[5]:.3f}\tAUROC:{auroc:.3f}\tBest_F1:{max_f1:.3f}(Thr:{best_thr:.3f})\tF1@0.5:{f1:.3f}\tR:{pearson_r:.3f}")
                        
                        if wandb_log:
                            wandb.log({
                                f"{data_set_name}/MAE": current_meters[5],
                                f"{data_set_name}/AUROC": auroc,
                                f"{data_set_name}/Max_F1": max_f1,
                                f"{data_set_name}/Best_Thr": best_thr,
                                f"{data_set_name}/F1_at_0.5": f1,
                                f"{data_set_name}/Pearson_R": pearson_r,
                            }, step=total_step)

                    with torch.no_grad():
                        evaluate_set(valid_loader, "Valid", total_step, wandb_log=True)
                        evaluate_set(test_loader, "Test", total_step, wandb_log=True)
                    PLANET_model.train()

            except Exception as e: continue
            
    wandb.finish()