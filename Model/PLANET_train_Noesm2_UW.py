# PLANET_train.py (DÜZENLENMİŞ)
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# --- PYTHON PATH GÜNCELLEMESİ (Modül Bulma Hatasını Çözer) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
datautils_path = os.path.join(current_dir, '..', 'Dataset_preparation')

if datautils_path not in sys.path:
    sys.path.append(datautils_path)
# -----------------------------------------------------------

# --- KRİTİK PROJE İÇİ IMPORT'LAR VE HATA YÖNETİMİ ---
try:
    from PLANET_datautils import ProLigDataset 
    from PLANET_model import PLANET 
    from chemutils import ComplexPocket,tensorize_all,role_of_5
    
    RDLogger.DisableLog('rdApp.*') 
    
except ImportError as e:
    print(f"KRİTİK HATA: Modül bulunamadı. Detay: {e}", file=sys.stderr)
    sys.exit(1)
# ----------------------------------------------------

# PyTorch'a özel Tensor taşıma yardımcı fonksiyonu
def move_to_device(data, device):
    """Veri yapısı içindeki tüm Tensor'ları hedef cihaza taşır."""
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

    # Model Parametreleri (Çift Atamadan kaçınıldı)
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
    parser.add_argument('--binder_per_batch', type=int, default=2)
    parser.add_argument('--load_checkpoint', default=None)
    # 🎯 YENİ ARGÜMAN: İnce Ayar Modu (Sadece Afinite Kaybı)
    parser.add_argument('--fine_tune_affinity', action='store_true', 
                        help="Etkinleştirilirse, geri yayılımda sadece Affinity Loss kullanılır.")


    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True) 
    print(args)

    # 1. WANDB BAŞLATMA
    wandb.init(
        project="PLANET_TRAINING",
        name=f"Noesm2_Uw{args.batch_size}",
        config=args,
        id=os.getenv('WANDB_RUN_ID'),
        resume=os.getenv('WANDB_RESUME'),
        entity=os.getenv('WANDB_ENTITY'),
        resume_from=os.getenv('WANDB_RESUME_FROM')
    )

    # 1. Cihaz Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim Cihazı: {device}")
    
    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=lambda x:x[0]) 

    # Test Veri Yükleyicileri
    # decoy_flag=True olmalı ve num_workers=0 olmalı
    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=lambda x:x[0])

    # 🚨 DÜZELTME 3: train_dataset ve train_loader DÖNGÜ DIŞINA TAŞINDI
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

    # 3. Model Başlatma
    PLANET_model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                          args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device).to(device)
    
    # 4. Ağırlık Başlatma ve Optimizer
    for param in PLANET_model.parameters():
        # 🎯 Güçlendirilmiş Başlatma Mantığı
        if param.dim() == 1:
            # 1 boyutlu parametreler (çoğunlukla bias ve log_sigma'lar) için sıfır başlatma
            nn.init.constant_(param, 0) 
        elif param.dim() >= 2: 
            # Sadece 2 veya daha fazla boyutu olanlar için Xavier kullan
            nn.init.xavier_uniform_(param)
        # Not: 0 boyutlu skalar parametreler (log_sigma'lar dahil) bu döngüde atlanacak/zaten init'te 0.0 ayarlandı.
            
    print ("Model #Params: {:d}K".format(sum([x.nelement() for x in PLANET_model.parameters()]) // 1000))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET_model.parameters()), lr=args.lr)

    # Örneğin, her 20.000 adımda bir LR'ı %10 düşürmek için:
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.90)
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
            
            # --- YENİ DÜZENLEME (DataLoader Hatası İçin) ---
            try:
                if batch_data is None:
                    continue
                res_feature_batch, mol_feature_batch, targets = batch_data 
            except Exception as e:
                # Veri yükleme hatasını yakala (OSError, pickle hatası vb.)
                print(f"VERİ YÜKLEME HATASI. ATLANAN BATCH (Step: {total_step}). Detay: {type(e).__name__} - {e}", file=sys.stderr)
                sys.stderr.flush()
                continue
            # ---------------------------------------------
            
            # 🚨 DÜZELTME 1A: Eğitim Verilerini Cihaza Taşı
            res_feature_batch = move_to_device(res_feature_batch, device)
            mol_feature_batch = move_to_device(mol_feature_batch, device)
            targets = move_to_device(targets, device)
            
            optimizer.zero_grad()
            
            try:
                # 3. Model Çağrısı
                predictions = PLANET_model(res_feature_batch, mol_feature_batch)
                
                # 4. Kayıp Hesaplama
                lig_interaction_loss,pro_lig_interaction_loss,affinity_loss = PLANET_model.compute_loss(predictions,targets,res_feature_batch,mol_feature_batch)
                
                # Eski (Manuel Ölçekleme ve Basit Toplama) Bloğu:
                # ----------------------------------------------------
                # AFFINITY_SCALE = 0.001
                # total_loss = lig_interaction_loss + pro_lig_interaction_loss + beta * affinity_loss * AFFINITY_SCALE
                # ----------------------------------------------------


                # 🎯 YENİ (UNÇERTAİNTY WEİGHTİNG) KAYIP HESAPLAMA BLOĞU
                # L_total = (L_i / (2 * sigma_i^2)) + log(sigma_i)

                # 1. Kayıp Ağırlıklarını Hesapla (exp(-log_sigma) = 1 / sigma^2)
                # Not: Ligand kayıplarının ağırlıklandırılması, makalede belirtilen L_total formülündeki tüm kayıplar için yapılmalıdır.
                # Affinity kaybında beta çarpanını koruyoruz, çünkü modelin erken aşamada regresyonu öğrenmesini önlemek istiyoruz.

                term_lig = torch.exp(-PLANET_model.log_sigma_lig) * lig_interaction_loss
                term_pl = torch.exp(-PLANET_model.log_sigma_pl) * pro_lig_interaction_loss
                term_affinity = torch.exp(-PLANET_model.log_sigma_affinity) * beta * affinity_loss 

                # 2. Log Sigma Terimlerini Ekle (Belirsizlik Cezası)
                # Bu terimler modelin sigma değerlerini sonsuza kadar düşürmesini engeller.
                log_term = PLANET_model.log_sigma_lig + PLANET_model.log_sigma_pl + PLANET_model.log_sigma_affinity

                # 3. Toplam Kaybı Hesapla (Tüm terimleri topla)
                total_loss = term_lig + term_pl + term_affinity + log_term
                
                total_loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, PLANET_model.parameters()), args.clip_norm)
                optimizer.step()
                
                # 5. Metrikler için Kayıp
                lig_interaction_loss,pro_lig_interaction_loss,affinity_loss = PLANET_model.compute_loss(predictions, targets, res_feature_batch, mol_feature_batch)
                lig_interaction_acc,pro_lig_interaction_acc,affinity_mae = PLANET.compute_metrics(predictions,targets) 
                
                # 6. Metre ve Adım Güncelleme
                meters = meters + np.array([lig_interaction_loss.item(),lig_interaction_acc.item(),pro_lig_interaction_loss.item(),\
                    pro_lig_interaction_acc.item(),affinity_loss.item(),affinity_mae.item()])
                total_step += 1

            except Exception as e:
                # Hata yakalandı (Bozuk kayıtlar bu yüzden atlanıyor)
                print(f"HATA YAKALANDI ve ATLANDI (Step: {total_step}). Detay: {type(e).__name__} - {e}", file=sys.stderr)
                sys.stderr.flush()
                continue
            
            
            # F. Beta Zamanlaması
            if total_step <= 500: beta = 0.0
            elif total_step <= args.anneal_iter: beta = (total_step - 500) / (args.anneal_iter - 500)
            else: beta = 1.0
            
            # G. Learning Rate Scheduler
            if total_step > 0 and total_step % 50000 == 0:
                for pg in optimizer.param_groups: pg['lr'] *= 0.95
                print(f"📉 LR Düşürüldü: {optimizer.param_groups[0]['lr']:.8f}")
                    
                    
           # --- 1. EĞİTİM LOGLARI (Standardized & Universal) ---
            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(f"[{total_step}] L_Loss:{meters[0]:.3f} L_Acc:{meters[1]:.3f} PL_Loss:{meters[2]:.3f} PL_Acc:{meters[3]:.3f} Aff_L:{meters[4]:.3f} MAE:{meters[5]:.3f} Beta:{beta:.2f}")
                
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

                # Uncertainty Parametrelerini Logla (Eğer modelde varsa)
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
                # Model isimlendirmesini dinamik yapıyoruz
                model_save_name = "PLANET_ESM2" if "esm2" in args.train else "PLANET"
                torch.save(PLANET_model.state_dict(), f"{args.save_dir}/{model_save_name}.iter-{total_step}")

                def run_evaluation(loader, name, total_step, wandb_log=False):
                    all_prolig_true, all_prolig_pred = [], []
                    all_affinity_true, all_affinity_pred = [], []
                    eval_meters = np.zeros(6)
                    batch_count = 0

                    with torch.no_grad():
                        for batch_data in loader:
                            try:
                                if batch_data is None: continue
                                res_b, mol_b, tgts = batch_data
                                
                                res_b = move_to_device(res_b, device)
                                mol_b = move_to_device(mol_b, device)
                                tgts = move_to_device(tgts, device)
                                
                                preds = PLANET_model(res_b, mol_b)
                                l_l, pl_l, af_l = PLANET_model.compute_loss(preds, tgts, res_b, mol_b)
                                l_a, pl_a, af_mae = PLANET_model.compute_metrics(preds, tgts)
                                
                                all_prolig_pred.append(preds[1].detach().cpu())
                                all_prolig_true.append(tgts[1].detach().cpu())
                                all_affinity_pred.append(preds[2].detach().cpu())
                                all_affinity_true.append(tgts[2].detach().cpu())

                                eval_meters += np.array([l_l.item(), l_a.item(), pl_l.item(), pl_a.item(), af_l.item(), af_mae.item()])
                                batch_count += 1
                            except Exception: continue

                    if batch_count == 0: return

                    # Tensör Birleştirme
                    prolig_true = torch.cat(all_prolig_true).numpy()
                    prolig_pred = torch.cat(all_prolig_pred).numpy()
                    aff_true = torch.cat(all_affinity_true).numpy()
                    aff_pred = torch.cat(all_affinity_pred).numpy()
                    eval_meters /= batch_count

                    # Metrikler (SABİT İSİMLER - TÜM MODELLERDE AYNI)
                    p_corr, _ = pearsonr(aff_true.flatten(), aff_pred.flatten()) if len(aff_true) > 1 else (0, 0)
                    s_corr, _ = spearmanr(aff_true.flatten(), aff_pred.flatten()) if len(aff_true) > 1 else (0, 0)
                    auroc = roc_auc_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.5
                    auprc = average_precision_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.0
                    
                    prolig_pred_bin = (prolig_pred > 0.5).astype(int)
                    f1 = f1_score(prolig_true, prolig_pred_bin, zero_division=0)
                    prec = precision_score(prolig_true, prolig_pred_bin, zero_division=0)
                    rec = recall_score(prolig_true, prolig_pred_bin, zero_division=0)

                    print(f"[{name}_{total_step}] MAE:{eval_meters[5]:.3f} Pearson:{p_corr:.3f} Spearman:{s_corr:.3f} AUROC:{auroc:.3f} F1:{f1:.3f}")
                    
                    if wandb_log:
                        wandb.log({
                            # Kayıplar (Aynı Grafik Altında)
                            f"{name}/Loss_Ligand": eval_meters[0],
                            f"{name}/Loss_ProLig": eval_meters[2],
                            f"{name}/Loss_Affinity": eval_meters[4],
                            # Temas ve Ayırt Edicilik (Aynı Grafik Altında)
                            f"{name}/Contact_AUROC": auroc,
                            f"{name}/Contact_AUPRC": auprc,
                            f"{name}/Decoy_F1": f1,
                            f"{name}/Decoy_Precision": prec,
                            f"{name}/Decoy_Recall": rec,
                            # Bağlanma Gücü (Aynı Grafik Altında)
                            f"{name}/Affinity_MAE": eval_meters[5],
                            f"{name}/Affinity_Pearson": p_corr,
                            f"{name}/Affinity_Spearman": s_corr,
                        }, step=total_step)

                print("--- Validasyon Başlıyor ---")
                run_evaluation(valid_loader, "Valid", total_step, wandb_log=True)
                run_evaluation(test_loader, "Test", total_step, wandb_log=True)
                print("--- Validasyon Bitti ---")
                
                PLANET_model.train()

    wandb.finish()