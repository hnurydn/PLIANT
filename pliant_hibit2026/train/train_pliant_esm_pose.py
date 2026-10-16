import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse, os, sys, random
import numpy as np
import pickle
from rdkit import RDLogger
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
import margin_losses   # the two margin losses

# ═══════════════════════════════════════════════════════════════════════════
# PLANET_train_esm2_vector_gating_normalized_seeded.py
# Copy of PLANET's seeded vector-gating training script for the NORMALISATION experiment.
# (2026-08-21). Tek fark: model importu layers_esm2_vector_gating_normalized
# It points at the normalised model module. The two normalisations come from the
# gate-diagnostic / geo_bias-diagnostic finding: the content-attention logit is scaled by
# sqrt(feature_dims) and geo_bias is bounded with tanh (see the header of the layers file).
# The data, optimisation and loss logic, the seeding and the resume-aware wandb handling are
# all unchanged from the original seeded script.
# ═══════════════════════════════════════════════════════════════════════════

scripts_path = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts"
normalized_path = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/normalization_experiment"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
if normalized_path not in sys.path:
    sys.path.insert(0, normalized_path)

# ---- 2026-09-17: DUZELTILMIS hat HER ZAMAN en onde olsun ----------------
# Yukaridaki sabit sys.path.insert(0, Scripts) eski/hatali PLANET_datautils*
# dosyalarini one aliyordu ve PYTHONPATH gecersiz kaliyordu.
_FIXED = "/arf/scratch/hnaydin/PLANET_FIXED_20260917"
for _p in (_FIXED + "/model", _FIXED + "/pipeline"):
    if os.path.isdir(_p):
        if _p in sys.path: sys.path.remove(_p)
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

try:
    # [22 Eyl 2026 -- KAPI DENEYI] cok cepli/ORJ parti + kol secimi (--model_module)
    import importlib
    from PLANET_datautils_esm2_vector_cokcep import ProLigDataset
    import PLANET_datautils_esm2_vector as _DU_ESKI   # kept for equivalence / backward compatibility
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

    # -- added: seed and max_step ------------------------------------------─
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_step', type=int, default=None)
    # ── kapi deneyi ──
    parser.add_argument('--model_module', default='PLANET_model_esm2_vector_gating_normalized_sumln_gn')
    parser.add_argument('--n_pocket', type=int, default=1)
    parser.add_argument('--n_decoy', type=int, default=11)
    parser.add_argument('--min_decoy', type=int, default=11)
    parser.add_argument('--orj', action='store_true')
    parser.add_argument('--olcek', type=float, default=1.0)
    parser.add_argument('--sik_adim', type=int, default=5000, help='bu adima kadar sik ckpt + sik kapi kaydi')
    parser.add_argument('--sik_ckpt', type=int, default=250)
    # ── 25 Eyl: POZ MARJ KAYBI (afinite kanalina goreli kisit) ──
    parser.add_argument('--marj_lambda', type=float, default=0.0,
                        help='0 = kapali (orijinal davranis). Acmak icin orn. 1.0')
    parser.add_argument('--marj', type=float, default=1.0, help='pK biriminde marj')
    parser.add_argument('--marj_oteleme_min', type=float, default=1.0)
    parser.add_argument('--marj_oteleme_max', type=float, default=3.0)
    parser.add_argument('--marj_her_n', type=int, default=1,
                        help='marj terimini her n adimda bir uygula (maliyet ayari)')
    parser.add_argument('--marj_baslangic', type=int, default=0,
                        help='bu adimdan once marj uygulanmaz')
    # -- MOLECULE MARGIN: within a pocket, y(binder) >= y(decoy) + margin (no extra forward pass) --
    parser.add_argument('--mol_marj_lambda', type=float, default=0.0)
    parser.add_argument('--mol_marj', type=float, default=1.0)
    # ── 28 Eyl: ZOR DECOY SECIMI (v4kont'taki gibi; verilmezse davranis DEGISMEZ) ──
    parser.add_argument('--zor_dosya', default=None, help='<split>_zorluk.pkl: {anahtar: decoy indeksleri, en zor once}')
    parser.add_argument('--zor_ust', type=int, default=None, help='ilk K zor icinden rastgele; None = en zor')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # -- added: seeding -----------------------------------------------------─────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"🎲 Seed sabitlendi: {args.seed}")

    # -- YENI: resume-aware wandb (SAVE_DIR'de run-id sakla, walltime kesintisinde
    # resume the same run, to avoid fragmenting it or having to merge runs by hand) --
    _wandb_id_file = os.path.join(args.save_dir, "wandb_run_id.txt")
    if os.path.exists(_wandb_id_file):
        with open(_wandb_id_file) as _f:
            _wandb_run_id = _f.read().strip()
        wandb.init(project="PLANET_AGG_ABLATION", name=f"KAPI_{os.environ.get('KOL_ADI','kol')}_seed{args.seed}",
                   id=_wandb_run_id, resume="allow")
    else:
        wandb.init(project="PLANET_AGG_ABLATION", name=f"KAPI_{os.environ.get('KOL_ADI','kol')}_seed{args.seed}")
        with open(_wandb_id_file, "w") as _f:
            _f.write(wandb.run.id)

    PLANET = importlib.import_module(args.model_module).PLANET
    S = args.olcek
    WARM = max(1, int(round(500 / S))); LR_DECAY = max(1, int(round(50000 / S)))
    print(f"[KAPI DENEYI] model={args.model_module} n_pocket={args.n_pocket} n_decoy={args.n_decoy} orj={args.orj} "
          f"olcek={S} | isinma={WARM} anneal={args.anneal_iter} lr_decay={LR_DECAY} save={args.save_iter} "
          f"print={args.print_iter} sik(<{args.sik_adim}) her {args.sik_ckpt} | max_step={args.max_step}", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Eğitim Cihazı: {device}")

    valid_dataset = _DU_ESKI.ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
    test_dataset = _DU_ESKI.ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)

    train_dataset = ProLigDataset(
        args.train, args.batch_size, shuffle=True, decoy_flag=True,
        n_pocket=args.n_pocket, n_decoy=args.n_decoy, min_decoy=args.min_decoy, orj=args.orj,
        zor_dosya=args.zor_dosya, zor_ust=args.zor_ust
    )

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])

    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=12, pin_memory=True,
        prefetch_factor=8, persistent_workers=True, shuffle=True, collate_fn=lambda x: x[0]
    )

    print(f"   Train: {len(train_dataset)} batch | Valid: {len(valid_dataset)} batch | Test: {len(test_dataset)} batch")

    PLANET_model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                           args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters, device).to(device)

    print(f"Model #Params: {sum([x.nelement() for x in PLANET_model.parameters()]) // 1000}K")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET_model.parameters()), lr=args.lr)

    if args.load_checkpoint:
        print(f"✅ Checkpoint Yükleniyor: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            PLANET_model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("✅ Optimizer durumu da yüklendi (LR + momentum korunuyor).")
        else:
            # -- FIX (2026-08-15): old-format checkpoints stored only the model weights, so the
            # optimizer and the learning rate were silently reset on resume. The LR is corrected
            # from initial_step to make up for the decay steps that were lost --
            PLANET_model.load_state_dict(ckpt)
            n_missed_decays = args.initial_step // 50000
            corrected_lr = args.lr * (0.95 ** n_missed_decays)
            for pg in optimizer.param_groups:
                pg['lr'] = corrected_lr
            print(f"⚠️ Eski format checkpoint (optimizer state yok, momentum sıfırlandı). "
                  f"LR {n_missed_decays} kaçırılan decay'e göre {corrected_lr:.8f} olarak düzeltildi.")
    else:
        for param in PLANET_model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param)

        # ---- [SUM+LN] SADECE ProteinEGNN.msg_norm'u duzelt -------------------
        # Yukaridaki global dongu dim()==1 olan HER parametreyi sifirliyor;
        # LayerNorm.weight also becomes 0 and the layer goes completely silent. The model must
        # var olan LayerNorm'larina DOKUNMUYORUZ (yayinlanmis modellerle
        # for comparability the only change should be msg_norm).
        _n_ln = 0
        for _name, _mod in PLANET_model.named_modules():
            if isinstance(_mod, nn.LayerNorm):   # 2026-09-17: res_embedding dahil HEPSI
                nn.init.constant_(_mod.weight, 1.0)
                nn.init.constant_(_mod.bias, 0.0)
                _n_ln += 1
        _tot_ln = sum(1 for _m in PLANET_model.modules() if isinstance(_m, nn.LayerNorm))
        print('[LN] TUM LayerNorm weight=1/bias=0: %d adet (toplam %d) — res_embedding DAHIL.' % (_n_ln, _tot_ln), flush=True)
        assert _n_ln == _tot_ln and _n_ln >= 3, 'LayerNorm onarimi eksik: %d/%d' % (_n_ln, _tot_ln)
        # ---------------------------------------------------------------------

    total_step = args.initial_step
    meters = np.zeros(6)
    # -- margin counters and a reproducible generator --
    marj_gen = torch.Generator(device=device); marj_gen.manual_seed(args.seed + 12345)
    marj_meters = np.zeros(3); marj_n = 0
    mm_meters = np.zeros(3); mm_n = 0
    if args.mol_marj_lambda > 0:
        print('[MOL MARJ] acik: lambda=%.3f marj=%.2f (binder vs decoy, ayni cep)'
              % (args.mol_marj_lambda, args.mol_marj), flush=True)
    if args.marj_lambda > 0:
        print('[MARJ] acik: lambda=%.3f marj=%.2f oteleme=%.1f-%.1f A her_n=%d baslangic=%d'
              % (args.marj_lambda, args.marj, args.marj_oteleme_min, args.marj_oteleme_max,
                 args.marj_her_n, args.marj_baslangic), flush=True)
    beta = 0.0
    stop_training = False  # added

    PLANET_model.train()
    print("🚀 Eğitim Başlıyor...")

    for epoch in range(1, 1 + args.epoch):
        train_dataset.reshuffle()
        for batch_data in train_loader:

            try:
                if batch_data is None: continue
                res_feature_batch, mol_feature_batch, targets = batch_data
            except Exception:
                continue

            res_feature_batch = move_to_device(res_feature_batch, device)
            mol_feature_batch = move_to_device(mol_feature_batch, device)
            targets = move_to_device(targets, device)

            optimizer.zero_grad()

            try:
                predictions = PLANET_model(res_feature_batch, mol_feature_batch)
                lig_loss, pl_loss, aff_loss = PLANET_model.compute_loss(
                    predictions, targets, res_feature_batch, mol_feature_batch)

                log_sigma_lig_c = torch.clamp(PLANET_model.log_sigma_lig, min=-5, max=5)
                log_sigma_pl_c = torch.clamp(PLANET_model.log_sigma_pl, min=-5, max=5)
                log_sigma_aff_c = torch.clamp(PLANET_model.log_sigma_affinity, min=-5, max=5)

                loss_lig_w = torch.exp(-log_sigma_lig_c) * lig_loss + log_sigma_lig_c
                loss_pl_w = torch.exp(-log_sigma_pl_c) * pl_loss + log_sigma_pl_c
                loss_aff_w = torch.exp(-log_sigma_aff_c) * (beta * aff_loss) + log_sigma_aff_c

                total_loss = loss_lig_w + loss_pl_w + loss_aff_w

                # -- POSE MARGIN: the correct pose must score at least `marj` better than the perturbed one --
                if (args.marj_lambda > 0 and total_step >= args.marj_baslangic
                        and total_step % args.marj_her_n == 0):
                    m_loss, m_fark, m_ihlal = margin_losses.pose_margin_loss(
                        PLANET_model, res_feature_batch, mol_feature_batch, targets,
                        marj_gen, predictions[2], marj=args.marj,
                        oteleme_min=args.marj_oteleme_min, oteleme_max=args.marj_oteleme_max)
                    total_loss = total_loss + args.marj_lambda * m_loss
                    marj_meters += np.array([m_loss.item(), m_fark.item(), m_ihlal.item()])
                    marj_n += 1

                if args.mol_marj_lambda > 0:
                    mm_loss, mm_fark, mm_ihlal = margin_losses.molecule_margin_loss(
                        predictions[2], targets, marj=args.mol_marj)
                    total_loss = total_loss + args.mol_marj_lambda * mm_loss
                    mm_meters += np.array([mm_loss.item(), mm_fark.item(), mm_ihlal.item()])
                    mm_n += 1

                total_loss.backward()
                nn.utils.clip_grad_norm_(PLANET_model.parameters(), args.clip_norm)
                optimizer.step()

                lig_acc, pl_acc, aff_mae = PLANET.compute_metrics(predictions, targets)
                meters += np.array([lig_loss.item(), lig_acc.item(), pl_loss.item(),
                                     pl_acc.item(), aff_loss.item(), aff_mae.item()])
                total_step += 1

            except Exception as e:
                print(f"⚠️ Eğitim Adımı Hatası: {e}")
                torch.cuda.empty_cache()
                continue

            if total_step <= WARM: beta = 0.0
            elif total_step <= args.anneal_iter: beta = (total_step - WARM) / (args.anneal_iter - WARM)
            else: beta = 1.0

            if total_step > 0 and total_step % LR_DECAY == 0:
                for pg in optimizer.param_groups: pg['lr'] *= 0.95
                print(f"📉 LR Düşürüldü: {optimizer.param_groups[0]['lr']:.8f}")

            # [KAPI DENEYI] fuzyon durumu: alpha (toplamali kollar) veya z (kapili kol)
            _sik = (total_step <= args.sik_adim and total_step % 100 == 0)
            if _sik or total_step % args.print_iter == 0:
                eg = PLANET_model.proteinegnn
                if hasattr(eg, 'alpha'):
                    _fz = f"alpha={eg.alpha.item():+.4f}"
                    _wd = {"Fuzyon/alpha": eg.alpha.item()}
                else:
                    with torch.no_grad():
                        _zs = []
                        _h = eg.gating_network.register_forward_hook(lambda m, i, o: _zs.append(o.detach().flatten()))
                        PLANET_model.eval(); PLANET_model(res_feature_batch, mol_feature_batch); PLANET_model.train(); _h.remove()
                        _z = torch.cat(_zs)
                    _fz = f"z_ort={_z.mean().item():.4f} z<0.01={100*(_z<0.01).float().mean().item():.1f}%"
                    _wd = {"Fuzyon/z_ort": _z.mean().item(), "Fuzyon/z_kapali_yuzde": 100*(_z<0.01).float().mean().item()}
                print(f"[FUZYON {total_step}] (esd {int(total_step*S)}) {_fz}", flush=True)
                try: wandb.log(_wd, step=total_step)
                except Exception: pass
            if total_step <= args.sik_adim and total_step % args.sik_ckpt == 0:
                torch.save({'model_state_dict': PLANET_model.state_dict(), 'total_step': total_step},
                           f"{args.save_dir}/PLANET.iter-{total_step}")
            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                _mm = marj_meters / max(marj_n, 1)
                _ms = (" Marj:%.3f Fark:%+.3f Ihlal:%.2f" % (_mm[0], _mm[1], _mm[2])) if marj_n else ""
                _m2 = mm_meters / max(mm_n, 1)
                _ms += (" mMarj:%.3f mFark:%+.3f mIhlal:%.2f" % (_m2[0], _m2[1], _m2[2])) if mm_n else ""
                print(f"[{total_step}] L_Loss:{meters[0]:.3f} L_Acc:{meters[1]:.3f} PL_Loss:{meters[2]:.3f} PL_Acc:{meters[3]:.3f} Aff_L:{meters[4]:.3f} MAE:{meters[5]:.3f} Beta:{beta:.2f}" + _ms)

                log_dict = {
                    "Epoch": epoch, "Total_Step": total_step, "Esdeger_Step": int(total_step * S),
                    "Train/Lig_Loss": meters[0], "Train/Lig_ACC": meters[1],
                    "Train/ProLig_Loss": meters[2], "Train/ProLig_ACC": meters[3],
                    "Train/Affinity_Loss": meters[4], "Train/MAE": meters[5],
                    "Hyperparams/Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Hyperparams/Beta": beta,
                }
                if marj_n:
                    log_dict.update({"Marj/Kayip": _mm[0], "Marj/Fark_pK": _mm[1],
                                     "Marj/Ihlal_orani": _mm[2]})
                if mm_n:
                    log_dict.update({"MolMarj/Kayip": _m2[0], "MolMarj/Fark_pK": _m2[1],
                                     "MolMarj/Ihlal_orani": _m2[2]})
                marj_meters *= 0; marj_n = 0; mm_meters *= 0; mm_n = 0
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
                torch.save({
                    'model_state_dict': PLANET_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_step': total_step,
                }, f"{args.save_dir}/{model_save_name}.iter-{total_step}")

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

                                eval_meters += np.array([l_l.item(), l_a.item(), pl_l.item(), pl_a.item(), af_l.item(), af_mae.item()])
                                batch_count += 1
                            except Exception:
                                continue

                    if batch_count == 0: return

                    prolig_true = torch.cat(all_prolig_true).numpy()
                    prolig_pred = torch.cat(all_prolig_pred).numpy()
                    aff_true = torch.cat(all_affinity_true).numpy()
                    aff_pred = torch.cat(all_affinity_pred).numpy()
                    eval_meters /= batch_count

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

                    print(f"[{name}_{total_step}] MAE:{eval_meters[5]:.3f} Pearson:{p_corr:.3f} Spearman:{s_corr:.3f} AUROC:{auroc:.3f} F1:{f1:.3f} EF0.5%:{ef05:.2f} EF1%:{ef1:.2f} EF2%:{ef2:.2f} EF5%:{ef5:.2f}")

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

                # -- added: max_step check --
                if args.max_step and total_step >= args.max_step:
                    print(f"🏁 max_step ({args.max_step}) ulaşıldı, checkpoint kaydedildi, çıkılıyor.")
                    stop_training = True
                    break

        if stop_training:
            break

    wandb.finish()
    print("✅ Eğitim tamamlandı (seed={}, son adım={}).".format(args.seed, total_step))
