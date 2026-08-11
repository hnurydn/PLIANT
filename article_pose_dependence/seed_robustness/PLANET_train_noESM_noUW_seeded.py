import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import argparse, os, sys, random
import numpy as np
from rdkit import RDLogger
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

# ═══════════════════════════════════════════════════════════════════════════
# PLANET_train_noESM_noUW_seeded.py
# Orijinal Scripts/PLANET_train_noESM_noUW.py'nin (GERÇEK üretim script'i --
# Model/ klasöründeki eski/farklı kopya DEĞİL) SEED-KONTROLLÜ + MAX_STEP'li
# kopyası. Degisiklikler: --seed + --max_step eklendi, geri kalan mantik
# birebir ayni.
# ═══════════════════════════════════════════════════════════════════════════

scripts_path = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from PLANET_datautils import ProLigDataset
    from PLANET_model import PLANET
    RDLogger.DisableLog('rdApp.*')
except ImportError as e:
    print(f"KRİTİK HATA: Modül bulunamadı. Detay: {e}", file=sys.stderr)
    sys.exit(1)


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return [move_to_device(t, device) for t in data]
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True)
    parser.add_argument('-v', '--valid', required=True)
    parser.add_argument('-te', '--test', required=True)
    parser.add_argument('-d', '--save_dir', required=True)

    parser.add_argument('-s', '--feature_dims', type=int, default=512)
    parser.add_argument('-n', '--nheads', type=int, default=8)
    parser.add_argument('-k', '--key_dims', type=int, default=128)
    parser.add_argument('-va', '--value_dims', type=int, default=128)
    parser.add_argument('-pu', '--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu', '--lig_update_iters', type=int, default=10)
    parser.add_argument('-pl', '--pro_lig_update_iters', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--binder_per_batch', type=int, default=1)
    parser.add_argument('--decoy_count', type=int, default=10)
    parser.add_argument('--anneal_iter', type=int, default=20000)
    parser.add_argument('--save_iter', type=int, default=20000)
    parser.add_argument('--print_iter', type=int, default=1000)
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--initial_step', type=int, default=0)

    # ── YENİ: seed + max_step ──────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_step', type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── YENİ: seed'leme ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"🎲 Seed sabitlendi: {args.seed}")

    wandb.init(project="PLANET_SEED_ROBUSTNESS", name=f"noesm_nouw_1b11d_seed{args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Eğitim Cihazı: {device}")

    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)

    train_dataset = ProLigDataset(
        args.train, args.batch_size, shuffle=True, decoy_flag=True,
        binder_per_batch=args.binder_per_batch
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=4, pin_memory=True,
        prefetch_factor=4, persistent_workers=True, shuffle=True, collate_fn=lambda x: x[0]
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])

    PLANET_model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                           args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device).to(device)

    for param in PLANET_model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif param.dim() >= 2:
            nn.init.xavier_uniform_(param)

    print("Model #Params: {:d}K".format(sum([x.nelement() for x in PLANET_model.parameters()]) // 1000))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET_model.parameters()), lr=args.lr)

    if args.load_checkpoint:
        print(f"✅ Checkpoint Yükleniyor: {args.load_checkpoint}")
        PLANET_model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    else:
        print("✅ Yeni model başlatılıyor.")

    total_step = args.initial_step
    meters = np.zeros(6)
    beta = 0
    stop_training = False  # YENİ

    PLANET_model.train()
    for epoch in range(1, 1 + args.epoch):

        train_dataset.reshuffle()

        for batch_data in train_loader:
            try:
                if batch_data is None:
                    continue
                res_feature_batch, mol_feature_batch, targets = batch_data
            except Exception as e:
                print(f"VERİ YÜKLEME HATASI (Step: {total_step}). Detay: {e}", file=sys.stderr)
                continue

            res_feature_batch = move_to_device(res_feature_batch, device)
            mol_feature_batch = move_to_device(mol_feature_batch, device)
            targets = move_to_device(targets, device)

            optimizer.zero_grad()

            try:
                predictions = PLANET_model(res_feature_batch, mol_feature_batch)
                lig_loss, pl_loss, aff_loss = PLANET_model.compute_loss(
                    predictions, targets, res_feature_batch, mol_feature_batch)

                total_loss = lig_loss + pl_loss + (beta * aff_loss)

                total_loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,
                                                 PLANET_model.parameters()), args.clip_norm)
                optimizer.step()

                lig_acc, pl_acc, aff_mae = PLANET.compute_metrics(predictions, targets)
                meters = meters + np.array([lig_loss.item(), lig_acc.item(), pl_loss.item(),
                                             pl_acc.item(), aff_loss.item(), aff_mae.item()])
                total_step += 1

            except Exception as e:
                print(f"⚠️ Eğitim Adımı Hatası: {e}", file=sys.stderr)
                sys.stderr.flush()
                continue

            if total_step <= 500: beta = 0.0
            elif total_step <= args.anneal_iter: beta = (total_step - 500) / (args.anneal_iter - 500)
            else: beta = 1.0

            if total_step > 0 and total_step % 50000 == 0:
                for pg in optimizer.param_groups: pg['lr'] *= 0.95
                print(f"📉 LR Düşürüldü: {optimizer.param_groups[0]['lr']:.8f}")

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(f"[{total_step}] L_Loss:{meters[0]:.3f} L_Acc:{meters[1]:.3f} PL_Loss:{meters[2]:.3f} PL_Acc:{meters[3]:.3f} Aff_L:{meters[4]:.3f} MAE:{meters[5]:.3f}")

                log_dict = {
                    "Epoch": epoch, "Total_Step": total_step,
                    "Train/Lig_Loss": meters[0], "Train/Lig_ACC": meters[1],
                    "Train/ProLig_Loss": meters[2], "Train/ProLig_ACC": meters[3],
                    "Train/Affinity_Loss": meters[4], "Train/MAE": meters[5],
                    "Hyperparams/Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Hyperparams/Beta": beta,
                }
                if hasattr(PLANET_model, 'log_sigma_lig'):
                    log_dict.update({
                        "Uncertainty/Sigma_Lig": torch.exp(PLANET_model.log_sigma_lig).item(),
                        "Uncertainty/Sigma_PL": torch.exp(PLANET_model.log_sigma_pl).item(),
                        "Uncertainty/Sigma_Affinity": torch.exp(PLANET_model.log_sigma_affinity).item(),
                    })
                wandb.log(log_dict, step=total_step)
                meters *= 0.

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
                                tgts = move_to_device(tgts, device)

                                preds = PLANET_model(res_b, mol_b)
                                l_l, pl_l, af_l = PLANET_model.compute_loss(preds, tgts, res_b, mol_b)
                                l_a, pl_a, af_mae = PLANET_model.compute_metrics(preds, tgts)

                                all_prolig_pred.append(preds[1].detach().cpu())
                                all_prolig_true.append(tgts[1].detach().cpu())
                                all_affinity_pred.append(preds[2].detach().cpu())
                                all_affinity_true.append(tgts[2].detach().cpu())
                                all_pK_flags.append(tgts[3].detach().cpu())

                                eval_meters += np.array([l_l.item(), l_a.item(), pl_l.item(),
                                                          pl_a.item(), af_l.item(), af_mae.item()])
                                batch_count += 1
                            except Exception:
                                continue

                    if batch_count == 0: return
                    eval_meters /= batch_count

                    prolig_true = torch.cat(all_prolig_true).numpy()
                    prolig_pred = torch.cat(all_prolig_pred).numpy()
                    aff_true = torch.cat(all_affinity_true).numpy()
                    aff_pred = torch.cat(all_affinity_pred).numpy()
                    pK_flags_np = torch.cat(all_pK_flags).numpy()

                    mask = pK_flags_np == 1.0
                    p_corr = pearsonr(aff_true[mask], aff_pred[mask])[0] if mask.sum() > 1 else 0.0
                    s_corr = spearmanr(aff_true[mask], aff_pred[mask])[0] if mask.sum() > 1 else 0.0
                    auroc = roc_auc_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.5
                    auprc = average_precision_score(prolig_true, prolig_pred) if np.sum(prolig_true) > 0 else 0.0

                    prolig_pred_bin = (prolig_pred > 0.5).astype(int)
                    f1 = f1_score(prolig_true, prolig_pred_bin, zero_division=0)
                    prec = precision_score(prolig_true, prolig_pred_bin, zero_division=0)
                    rec = recall_score(prolig_true, prolig_pred_bin, zero_division=0)

                    def calc_ef(active_scores, decoy_scores, pct):
                        import math
                        y_true = np.concatenate([np.ones(len(active_scores)), np.zeros(len(decoy_scores))])
                        y_score = np.concatenate([active_scores, decoy_scores])
                        sorted_labels = y_true[np.argsort(-y_score)]
                        n_total, n_actives = len(y_score), len(active_scores)
                        n_sample = math.ceil(n_total * pct)
                        if n_sample == 0 or n_actives == 0: return 0.0
                        return (sorted_labels[:n_sample].sum() / n_sample) / (n_actives / n_total)

                    active_aff = aff_pred[mask]
                    decoy_aff = aff_pred[~mask]
                    ef05 = calc_ef(active_aff, decoy_aff, 0.005)
                    ef1 = calc_ef(active_aff, decoy_aff, 0.01)
                    ef2 = calc_ef(active_aff, decoy_aff, 0.02)
                    ef5 = calc_ef(active_aff, decoy_aff, 0.05)

                    print(f"[{name}_{total_step}] MAE:{eval_meters[5]:.3f} Pearson:{p_corr:.3f} "
                          f"Spearman:{s_corr:.3f} AUROC:{auroc:.3f} F1:{f1:.3f} "
                          f"EF0.5%:{ef05:.2f} EF1%:{ef1:.2f} EF2%:{ef2:.2f} EF5%:{ef5:.2f}")

                    if wandb_log:
                        wandb.log({
                            f"{name}/Loss_Ligand": eval_meters[0],
                            f"{name}/Loss_ProLig": eval_meters[2],
                            f"{name}/Loss_Affinity": eval_meters[4],
                            f"{name}/Contact_AUROC": auroc,
                            f"{name}/Contact_AUPRC": auprc,
                            f"{name}/Contact_F1": f1,
                            f"{name}/Contact_Precision": prec,
                            f"{name}/Contact_Recall": rec,
                            f"{name}/Affinity_MAE": eval_meters[5],
                            f"{name}/Affinity_Pearson": p_corr,
                            f"{name}/Affinity_Spearman": s_corr,
                            f"{name}/VS_EF0.5": ef05,
                            f"{name}/VS_EF1": ef1,
                            f"{name}/VS_EF2": ef2,
                            f"{name}/VS_EF5": ef5,
                        }, step=total_step)

                print("--- Validasyon Başlıyor ---")
                run_evaluation(valid_loader, "Valid", total_step, wandb_log=True)
                run_evaluation(test_loader, "Test", total_step, wandb_log=True)
                print("--- Validasyon Bitti ---")

                PLANET_model.train()

                # ── YENİ: max_step kontrolü ──
                if args.max_step and total_step >= args.max_step:
                    print(f"🏁 max_step ({args.max_step}) ulaşıldı, checkpoint kaydedildi, çıkılıyor.")
                    stop_training = True
                    break

        if stop_training:
            break

    wandb.finish()
    print("✅ Eğitim tamamlandı (seed={}, son adım={}).".format(args.seed, total_step))
