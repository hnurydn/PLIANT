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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

# --- PYTHON PATH GÜNCELLEMESİ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
datautils_path = os.path.join(current_dir, '..', 'Dataset_preparation')

if datautils_path not in sys.path:
    sys.path.append(datautils_path)
# --------------------------------

# --- IMPORTLAR VE HATA YÖNETİMİ ---
try:
    from PLANET_datautils import ProLigDataset 
    from PLANET_model import PLANET 
    from chemutils import ComplexPocket, tensorize_all, role_of_5
    
    RDLogger.DisableLog('rdApp.*') 
    
except ImportError as e:
    print(f"KRİTİK HATA: Modül bulunamadı. Detay: {e}", file=sys.stderr)
    sys.exit(1)
# ----------------------------------

# PyTorch'a özel Tensor taşıma yardımcı fonksiyonu
def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return [move_to_device(t, device) for t in data]
    return data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Argümanlar
    parser.add_argument('-t','--train', required=True, help="Yol Train.pkl dosyasına.")
    parser.add_argument('-v','--valid',required=True, help="Yol Valid.pkl dosyasına.")
    parser.add_argument('-te','--test',required=True, help="Yol Core.pkl dosyasına.")
    parser.add_argument('-d','--save_dir', required=True, help="Model ağırlıklarının kaydedileceği dizin.")

    # Model Parametreleri
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)

    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--anneal_iter', type=int, default=50000)
    
    parser.add_argument('--initial_step',type=int,default=0)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--print_iter', type=int, default=1000)
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--binder_per_batch', type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True) 
    print(args)

    # 1. WANDB BAŞLATMA
    wandb.init(project="PLANET_TRAINING", name=f"Noesm2_NoUW{args.batch_size}", config=args)

    # 2. Cihaz Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim Cihazı: {device}")
    
    # 3. Veri Yükleyicileri (Dataset yapısı korundu)
    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=lambda x:x[0]) 

    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=lambda x:x[0])

    train_dataset = ProLigDataset(
        args.train, 
        args.batch_size, 
        shuffle=True, 
        decoy_flag=True, 
        binder_per_batch = args.binder_per_batch
    )
    # --- train_loader kısmını sadece şu şekilde güncellemen yeterli ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # ProLigDataset kendi içinde batch'liyor
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=True, # Veri düzeyinde rastgelelik
        collate_fn=lambda x: x[0]
    )

    # 4. Model Başlatma
    PLANET_model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                          args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device).to(device)
    
    # 5. Ağırlık Başlatma ve Optimizer
    for param in PLANET_model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0) 
        elif param.dim() >= 2: 
            nn.init.xavier_uniform_(param)
            
    print ("Model #Params: {:d}K".format(sum([x.nelement() for x in PLANET_model.parameters()]) // 1000))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET_model.parameters()), lr=args.lr)

    if args.load_checkpoint:
        print(f"✅ Checkpoint Yükleniyor: {args.load_checkpoint}")
        PLANET_model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    else:
        print("✅ Yeni model başlatılıyor.")
    
    total_step = args.initial_step
    meters = np.zeros(6)
    beta = 0
    
    # 6. Eğitim Döngüsü
    PLANET_model.train()
    for epoch in range(1, 1 + args.epoch):
        
        for batch_data in train_loader:
            
            # --- Veri Kontrolü ---
            try:
                if batch_data is None:
                    continue
                res_feature_batch, mol_feature_batch, targets = batch_data 
            except Exception as e:
                print(f"VERİ YÜKLEME HATASI (Step: {total_step}). Detay: {e}", file=sys.stderr)
                continue
            
            # --- Cihaza Taşıma ---
            res_feature_batch = move_to_device(res_feature_batch, device)
            mol_feature_batch = move_to_device(mol_feature_batch, device)
            targets = move_to_device(targets, device)
            
            optimizer.zero_grad()
            
            try:
                # A. Model Çağrısı
                predictions = PLANET_model(res_feature_batch, mol_feature_batch)
                
                # B. Kayıp Hesaplama (Ham kayıplar)
                lig_interaction_loss, pro_lig_interaction_loss, affinity_loss = PLANET_model.compute_loss(predictions, targets, res_feature_batch, mol_feature_batch)
                
                # ----------------------------------------------------------------
                # 🔥 DEĞİŞİKLİK BURADA: ORİJİNAL BETA VE LOSS MANTIĞI 🔥
                # ----------------------------------------------------------------
                
                # F. Beta Zamanlaması
                if total_step <= 500: beta = 0.0
                elif total_step <= args.anneal_iter: beta = (total_step - 500) / (args.anneal_iter - 500)
                else: beta = 1.0
                
                # G. Learning Rate Scheduler
                if total_step > 0 and total_step % 50000 == 0:
                    for pg in optimizer.param_groups: pg['lr'] *= 0.95
                    print(f"📉 LR Düşürüldü: {optimizer.param_groups[0]['lr']:.8f}")

                # 2. Toplam Kayıp (Basit Toplama - Uncertainty YOK)
                total_loss = lig_interaction_loss + pro_lig_interaction_loss + (beta * affinity_loss)
                
                # ----------------------------------------------------------------

                total_loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, PLANET_model.parameters()), args.clip_norm)
                optimizer.step()
                
                # C. Metrikler için Kayıp
                lig_interaction_acc, pro_lig_interaction_acc, affinity_mae = PLANET.compute_metrics(predictions, targets) 
                
                # D. Metre ve Adım Güncelleme
                meters = meters + np.array([lig_interaction_loss.item(), lig_interaction_acc.item(), pro_lig_interaction_loss.item(),
                                            pro_lig_interaction_acc.item(), affinity_loss.item(), affinity_mae.item()])
                total_step += 1

            except Exception as e:
                print(f"HATA YAKALANDI ve ATLANDI (Step: {total_step}). Detay: {e}", file=sys.stderr)
                sys.stderr.flush()
                continue
            
            # --- 1. EĞİTİM LOGLARI (Standardized & Universal) ---
            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(f"[{total_step}] L_Loss:{meters[0]:.3f} L_Acc:{meters[1]:.3f} PL_Loss:{meters[2]:.3f} PL_Acc:{meters[3]:.3f} Aff_L:{meters[4]:.3f} MAE:{meters[5]:.3f}")
                
                log_dict = {
                    "Epoch": epoch,
                    "Total_Step": total_step,
                    "Train/Lig_Loss": meters[0],
                    "Train/Lig_ACC": meters[1],
                    "Train/ProLig_Loss": meters[2],
                    "Train/ProLig_ACC": meters[3],
                    "Train/Affinity_Loss": meters[4],
                    "Train/MAE": meters[5],
                    "Hyperparams/Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Hyperparams/Beta": beta,
                }

                # 🎯 Uncertainty Kontrolü: noUW modelinde hasattr False dönecek ve hata vermeyecektir.
                # UW olan modellerde ise otomatik olarak sigma grafiklerini oluşturacaktır.
                if hasattr(PLANET_model, 'log_sigma_lig'):
                    log_dict.update({
                        "Uncertainty/Sigma_Lig": torch.exp(PLANET_model.log_sigma_lig).item(),
                        "Uncertainty/Sigma_PL": torch.exp(PLANET_model.log_sigma_pl).item(),
                        "Uncertainty/Sigma_Affinity": torch.exp(PLANET_model.log_sigma_affinity).item(),
                    })
                
                wandb.log(log_dict, step=total_step)
                meters *= 0. 

            # --- 2. VALIDASYON VE TEST MODÜLÜ (Universal Version) ---
            if total_step > 0 and total_step % args.save_iter == 0:
                PLANET_model.eval()
                # Model isimlendirmesini dinamik yapıyoruz (ESM2 veya Standart için uyumlu)
                model_save_name = "PLANET_ESM2" if "esm2" in args.train else "PLANET"
                torch.save(PLANET_model.state_dict(), f"{args.save_dir}/{model_save_name}.iter-{total_step}")

                def evaluate_set(data_loader, data_set_name, total_step, wandb_log=False):
                    all_prolig_true, all_prolig_pred = [], []
                    all_affinity_true, all_affinity_pred = [], []
                    current_meters = np.zeros(6)
                    batch_count = 0 

                    with torch.no_grad():
                        for batch_data in data_loader:
                            try:
                                if batch_data is None: continue
                                res_batch, mol_batch, targets = batch_data
                                
                                res_batch = move_to_device(res_batch, device)
                                mol_batch = move_to_device(mol_batch, device)
                                targets = move_to_device(targets, device)
                                
                                preds = PLANET_model(res_batch, mol_batch) 
                                l_l, pl_l, af_l = PLANET_model.compute_loss(preds, targets, res_batch, mol_batch)
                                l_a, pl_a, af_mae = PLANET_model.compute_metrics(preds, targets)
                                
                                # Verileri CPU'ya alarak biriktir (OOM hatasını önlemek için)
                                all_prolig_pred.append(preds[1].detach().cpu())
                                all_prolig_true.append(targets[1].detach().cpu())
                                all_affinity_pred.append(preds[2].detach().cpu())
                                all_affinity_true.append(targets[2].detach().cpu())

                                current_meters += np.array([l_l.item(), l_a.item(), pl_l.item(), pl_a.item(), af_l.item(), af_mae.item()])
                                batch_count += 1
                            except Exception: continue

                    if batch_count == 0: return 

                    # Tensör Birleştirme
                    prolig_true = torch.cat(all_prolig_true).numpy()
                    prolig_pred = torch.cat(all_prolig_pred).numpy()
                    aff_true = torch.cat(all_affinity_true).numpy()
                    aff_pred = torch.cat(all_affinity_pred).numpy()
                    current_meters /= batch_count

                    # Metrik Hesaplamaları
                    p_corr, _ = pearsonr(aff_true.flatten(), aff_pred.flatten()) if len(aff_true) > 1 else (0, 0)
                    s_corr, _ = spearmanr(aff_true.flatten(), aff_pred.flatten()) if len(aff_true) > 1 else (0, 0)
                    auroc = roc_auc_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.5
                    auprc = average_precision_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.0
                    
                    prolig_pred_bin = (prolig_pred > 0.5).astype(int)
                    f1 = f1_score(prolig_true, prolig_pred_bin, zero_division=0)
                    prec = precision_score(prolig_true, prolig_pred_bin, zero_division=0)
                    rec = recall_score(prolig_true, prolig_pred_bin, zero_division=0)

                    if wandb_log:
                        wandb.log({
                            # Kayıplar
                            f"{data_set_name}/Loss_Ligand": current_meters[0],
                            f"{data_set_name}/Loss_ProLig": current_meters[2],
                            f"{data_set_name}/Loss_Affinity": current_meters[4],
                            # Temas ve Ayırt Edicilik
                            f"{data_set_name}/Contact_AUROC": auroc,
                            f"{data_set_name}/Contact_AUPRC": auprc,
                            f"{data_set_name}/Decoy_F1": f1,
                            f"{data_set_name}/Decoy_Precision": prec,
                            f"{data_set_name}/Decoy_Recall": rec,
                            # Bağlanma Gücü (Afinite)
                            f"{data_set_name}/Affinity_MAE": current_meters[5],
                            f"{data_set_name}/Affinity_Pearson": p_corr,
                            f"{data_set_name}/Affinity_Spearman": s_corr,
                        }, step=total_step)

                # Validasyon ve Testi Çalıştır
                evaluate_set(valid_loader, "Valid", total_step, wandb_log=True)
                evaluate_set(test_loader, "Test", total_step, wandb_log=True)
                PLANET_model.train()
        
    wandb.finish()