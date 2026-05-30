import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse, os, sys
import numpy as np
from rdkit import RDLogger
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

# --- İMPORT AYARLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
datautils_path = current_dir
if datautils_path not in sys.path:
    sys.path.append(datautils_path)

try:
    from PLANET_datautils_esm2_vector import ProLigDataset          # ← LMDB versiyonu
    from PLANET_model_esm2_vector_gating_contrastive_loss_v4 import PLANET
    RDLogger.DisableLog('rdApp.*')
except ImportError as e:
    print(f"KRİTİK HATA: Modül bulunamadı. Detay: {e}", file=sys.stderr)
    sys.exit(1)

# --- YARDIMCI FONKSİYONLAR ---
def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return [move_to_device(t, device) for t in data]
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dosya Yolları (LMDB klasör yolları)
    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
    parser.add_argument('-d',  '--save_dir', required=True, help="Model kayıt dizini")

    # Model Hiperparametreleri
    parser.add_argument('-s',  '--feature_dims',         type=int, default=512)
    parser.add_argument('-n',  '--nheads',               type=int, default=8)
    parser.add_argument('-k',  '--key_dims',             type=int, default=128)
    parser.add_argument('-va', '--value_dims',           type=int, default=128)
    parser.add_argument('-pu', '--pro_update_inters',    type=int, default=3)
    parser.add_argument('-lu', '--lig_update_iters',     type=int, default=10)
    parser.add_argument('-pl', '--pro_lig_update_iters', type=int, default=1)

    # Eğitim Ayarları
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--clip_norm',        type=float, default=1.0)
    parser.add_argument('--epoch',            type=int,   default=250)
    parser.add_argument('--batch_size',       type=int,   default=12)
    parser.add_argument('--binder_per_batch', type=int,   default=1,
                        help="Her batch'te kaç pocket: toplam molekül = binder_per_batch * 12")
    parser.add_argument('--decoy_count',      type=int,   default=10,
                        help="Valid/test'te her binder için max kaç decoy alınsın")
    parser.add_argument('--anneal_iter',      type=int,   default=20000)
    parser.add_argument('--save_iter',        type=int,   default=20000)
    parser.add_argument('--print_iter',       type=int,   default=1000)
    parser.add_argument('--load_checkpoint',  default=None)
    parser.add_argument('--initial_step',     type=int,   default=0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(
        project="PLANET_TRAINING",
        name=f"esm_uw_vg_contrastive_v4_projection_scratch"
    )

    # 2. CİHAZ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Eğitim Cihazı: {device}")

    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
    
    train_dataset = ProLigDataset(
        args.train,
        args.batch_size,
        shuffle=True,
        decoy_flag=True,
        binder_per_batch=args.binder_per_batch
    )

    # DataLoaders
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0]) 
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,        # 5 → 12 (cpus-per-task=20, 12 worker + 4 eval + 4 ana process)
        pin_memory=True,       # False → True (CPU→GPU transfer hızlanır)
        prefetch_factor=4,     # 4 → 8 (daha fazla batch önceden hazırlanır)
        persistent_workers=True,  # False → True (worker'lar epoch sonunda ölmesin)
        shuffle=True,
        collate_fn=lambda x: x[0]
    )

    print(f"   Train: {len(train_dataset)} batch | Valid: {len(valid_dataset)} batch | Test: {len(test_dataset)} batch")

    # 5. MODEL
    PLANET_model = PLANET(
        args.feature_dims, args.nheads, args.key_dims, args.value_dims,
        args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device
    ).to(device)

    for param in PLANET_model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif param.dim() >= 2:
            nn.init.xavier_uniform_(param)

    print(f"Model #Params: {sum(x.nelement() for x in PLANET_model.parameters()) // 1000}K")

    # 6. OPTİMİZER
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, PLANET_model.parameters()), lr=args.lr
    )

    if args.load_checkpoint:
        print(f"✅ Checkpoint Yükleniyor: {args.load_checkpoint}")
        PLANET_model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))

    total_step = args.initial_step
    meters     = np.zeros(6)
    con_loss_accum = 0.0
    beta       = 0.0

    # --------------------------------------------------------------------------
    # EĞİTİM DÖNGÜSÜ
    # --------------------------------------------------------------------------
    PLANET_model.train()
    print("🚀 Eğitim Başlıyor...")

    for epoch in range(1, 1 + args.epoch):

        train_dataset.reshuffle()   # ← Her epoch yeni batch kompozisyonu

        for batch_data in train_loader:
            try:
                if batch_data is None: continue
                res_feature_batch, mol_feature_batch, targets = batch_data
                res_feature_batch = move_to_device(res_feature_batch, device)
                mol_feature_batch = move_to_device(mol_feature_batch, device)
                targets           = move_to_device(targets, device)

                optimizer.zero_grad()
                predictions = PLANET_model(res_feature_batch, mol_feature_batch)
                lig_loss, pl_loss, aff_loss, con_loss = PLANET_model.compute_loss(
                    predictions, targets, res_feature_batch, mol_feature_batch
                )

                # Uncertainty Weighting — v4: contrastive ayrı sigma
                log_sigma_lig_c = torch.clamp(PLANET_model.log_sigma_lig,         min=-2, max=2)
                log_sigma_pl_c  = torch.clamp(PLANET_model.log_sigma_pl,          min=-2, max=2)
                log_sigma_aff_c = torch.clamp(PLANET_model.log_sigma_affinity,    min=-2, max=2)
                log_sigma_con_c = torch.clamp(PLANET_model.log_sigma_contrastive, min=-2, max=2)

                loss_lig_w = torch.exp(-log_sigma_lig_c) * lig_loss + log_sigma_lig_c
                loss_pl_w  = torch.exp(-log_sigma_pl_c)  * pl_loss  + log_sigma_pl_c
                loss_aff_w = torch.exp(-log_sigma_aff_c) * (beta * aff_loss) + log_sigma_aff_c
                # v4: contrastive kendi sigma'sıyla, bağımsız
                loss_con_w = torch.exp(-log_sigma_con_c) * (beta * con_loss) + log_sigma_con_c if beta > 0 else 0.0

                total_loss = loss_lig_w + loss_pl_w + loss_aff_w + loss_con_w
                total_loss.backward()
                nn.utils.clip_grad_norm_(PLANET_model.parameters(), args.clip_norm)
                optimizer.step()

                lig_acc, pl_acc, aff_mae = PLANET.compute_metrics(predictions, targets)
                meters += np.array([lig_loss.item(), lig_acc.item(), pl_loss.item(),
                                    pl_acc.item(), aff_loss.item(), aff_mae.item()])
                con_loss_accum += con_loss.item()
                total_step += 1

            except Exception as e:
                torch.cuda.empty_cache()
                continue

            if   total_step <= 500:              beta = 0.0
            elif total_step <= args.anneal_iter: beta = (total_step - 500) / (args.anneal_iter - 500)
            else:                                beta = 1.0
            
            # LR Scheduler
            if total_step > 0 and total_step % 50000 == 0:
                for pg in optimizer.param_groups: pg['lr'] *= 0.95
                print(f"📉 LR: {optimizer.param_groups[0]['lr']:.8f}")

            # Loglama
            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                avg_con = con_loss_accum / args.print_iter
                print(f"[{total_step}] L_Loss:{meters[0]:.3f} L_Acc:{meters[1]:.3f} "
                      f"PL_Loss:{meters[2]:.3f} PL_Acc:{meters[3]:.3f} "
                      f"Aff_L:{meters[4]:.3f} Con_L:{avg_con:.3f} MAE:{meters[5]:.3f} Beta:{beta:.2f}")

                log_dict = {
                    "Epoch": epoch, "Total_Step": total_step,
                    "Train/Lig_Loss":         meters[0], "Train/Lig_ACC":      meters[1],
                    "Train/ProLig_Loss":      meters[2], "Train/ProLig_ACC":   meters[3],
                    "Train/Affinity_Loss":    meters[4], "Train/MAE":          meters[5],
                    "Train/Contrastive_Loss": avg_con,
                    "Hyperparams/LR":   optimizer.param_groups[0]['lr'],
                    "Hyperparams/Beta": beta,
                }
                if hasattr(PLANET_model, 'log_sigma_lig'):
                    log_dict.update({
                        "Uncertainty/Sigma_Lig":         torch.exp(PLANET_model.log_sigma_lig).item(),
                        "Uncertainty/Sigma_PL":          torch.exp(PLANET_model.log_sigma_pl).item(),
                        "Uncertainty/Sigma_Affinity":    torch.exp(PLANET_model.log_sigma_affinity).item(),
                        "Uncertainty/Sigma_Contrastive": torch.exp(PLANET_model.log_sigma_contrastive).item(),
                    })
                wandb.log(log_dict, step=total_step)
                meters *= 0.
                con_loss_accum = 0.0


            # Validasyon & Test
            if total_step > 0 and total_step % args.save_iter == 0:
                PLANET_model.eval()
                model_save_name = "PLANET_ESM2" if "esm2" in args.train else "PLANET"
                torch.save(PLANET_model.state_dict(), f"{args.save_dir}/{model_save_name}.iter-{total_step}")

                def run_evaluation(loader, name, total_step, wandb_log=False):
                    all_prolig_true, all_prolig_pred = [], []
                    all_pK_flags = []
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
                                tgts  = move_to_device(tgts, device)

                                preds = PLANET_model(res_b, mol_b)
                                l_l, pl_l, af_l, con_l = PLANET_model.compute_loss(preds, tgts, res_b, mol_b)
                                l_a, pl_a, af_mae = PLANET_model.compute_metrics(preds, tgts)

                                all_prolig_pred.append(preds[1].detach().cpu())
                                all_prolig_true.append(tgts[1].detach().cpu())
                                all_affinity_pred.append(preds[2].detach().cpu())
                                all_affinity_true.append(tgts[2].detach().cpu())
                                all_pK_flags.append(tgts[3].detach().cpu())

                                eval_meters += np.array([l_l.item(), l_a.item(), pl_l.item(),
                                                        pl_a.item(), (af_l + con_l).item(), af_mae.item()])
                                batch_count += 1
                            except Exception as e:
                                print(f"EVAL ERROR: {e}")
                                continue

                    if batch_count == 0: return
                    eval_meters /= batch_count

                    prolig_true = torch.cat(all_prolig_true).numpy()
                    prolig_pred = torch.cat(all_prolig_pred).numpy()
                    aff_true    = torch.cat(all_affinity_true).numpy()
                    aff_pred    = torch.cat(all_affinity_pred).numpy()
                    pK_flags_np = torch.cat(all_pK_flags).numpy()

                    mask   = pK_flags_np == 1.0
                    p_corr = pearsonr(aff_true[mask],  aff_pred[mask])[0]  if mask.sum() > 1 else 0.0
                    s_corr = spearmanr(aff_true[mask], aff_pred[mask])[0]  if mask.sum() > 1 else 0.0
                    auroc  = roc_auc_score(prolig_true, prolig_pred)        if np.sum(prolig_true) > 0 else 0.5
                    auprc  = average_precision_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.0

                    prolig_pred_bin = (prolig_pred > 0.5).astype(int)
                    f1   = f1_score(prolig_true,        prolig_pred_bin, zero_division=0)
                    prec = precision_score(prolig_true, prolig_pred_bin, zero_division=0)
                    rec  = recall_score(prolig_true,    prolig_pred_bin, zero_division=0)

                    def calc_ef(active_scores, decoy_scores, pct):
                        import math
                        y_true  = np.concatenate([np.ones(len(active_scores)), np.zeros(len(decoy_scores))])
                        y_score = np.concatenate([active_scores, decoy_scores])
                        sorted_labels = y_true[np.argsort(-y_score)]
                        n_total, n_actives = len(y_score), len(active_scores)
                        n_sample = math.ceil(n_total * pct)
                        if n_sample == 0 or n_actives == 0: return 0.0
                        return (sorted_labels[:n_sample].sum() / n_sample) / (n_actives / n_total)

                    active_aff = aff_pred[mask]
                    decoy_aff  = aff_pred[~mask]
                    ef05 = calc_ef(active_aff, decoy_aff, 0.005)
                    ef1  = calc_ef(active_aff, decoy_aff, 0.01)
                    ef2  = calc_ef(active_aff, decoy_aff, 0.02)
                    ef5  = calc_ef(active_aff, decoy_aff, 0.05)

                    print(f"[{name}_{total_step}] MAE:{eval_meters[5]:.3f} Pearson:{p_corr:.3f} "
                        f"Spearman:{s_corr:.3f} AUROC:{auroc:.3f} F1:{f1:.3f} "
                        f"EF0.5%:{ef05:.2f} EF1%:{ef1:.2f} EF2%:{ef2:.2f} EF5%:{ef5:.2f}")

                    if wandb_log:
                        wandb.log({
                            f"{name}/Loss_Ligand":       eval_meters[0],
                            f"{name}/Loss_ProLig":       eval_meters[2],
                            f"{name}/Loss_Affinity":     eval_meters[4],
                            f"{name}/Contact_AUROC":     auroc,
                            f"{name}/Contact_AUPRC":     auprc,
                            f"{name}/Contact_F1":        f1,
                            f"{name}/Contact_Precision": prec,
                            f"{name}/Contact_Recall":    rec,
                            f"{name}/Affinity_MAE":      eval_meters[5],
                            f"{name}/Affinity_Pearson":  p_corr,
                            f"{name}/Affinity_Spearman": s_corr,
                            f"{name}/VS_EF0.5":          ef05,
                            f"{name}/VS_EF1":            ef1,
                            f"{name}/VS_EF2":            ef2,
                            f"{name}/VS_EF5":            ef5,
                        }, step=total_step)

                print("--- Validasyon Başlıyor ---")
                run_evaluation(valid_loader, "Valid", total_step, wandb_log=True)
                run_evaluation(test_loader,  "Test",  total_step, wandb_log=True)
                print("--- Validasyon Bitti ---")

                PLANET_model.train()

    wandb.finish()