import torch
import os, sys, argparse, re
import numpy as np
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, roc_curve)
from scipy.stats import pearsonr

# ─── Dinamik import: --model_type argümanına göre ─────────────────────────────
# argparse henüz parse edilmeden önce sys.argv'den model_type'ı çekiyoruz
# çünkü import'lar modül seviyesinde olması gerekiyor.
def _get_model_type_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--model_type' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_mt = _get_model_type_early()
if _mt is None:
    print("HATA: --model_type argümanı zorunludur.")
    print("Geçerli değerler: noesm | esm | vg | contrastive")
    sys.exit(1)

# Script'in bulunduğu dizini Python path'e ekle
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

if _mt == 'noesm':
    from PLANET_datautils import ProLigDataset
    from PLANET_model import PLANET
elif _mt == 'esm':
    from PLANET_datautils_esm2 import ProLigDataset
    from PLANET_model_esm2 import PLANET
elif _mt == 'vg':
    from PLANET_datautils_esm2_vector import ProLigDataset
    from PLANET_model_esm2_vector_gating import PLANET
elif _mt == 'contrastive':
    from PLANET_datautils_esm2_vector import ProLigDataset
    from PLANET_model_esm2_vector_gating_contrastive_loss import PLANET
else:
    print(f"HATA: Bilinmeyen model_type='{_mt}'")
    print("Geçerli değerler: noesm | esm | vg | contrastive")
    sys.exit(1)

print(f"✅ Model tipi yüklendi: {_mt}")

# ──────────────────────────────────────────────────────────────────────────────

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

def collect_predictions(model, loader, device, name):
    """Tüm tahminleri toplar, metrik hesaplamaz."""
    all_p_true_global, all_p_pred_global = [], []
    all_p_true_binder, all_p_pred_binder = [], []
    all_affinity_true, all_affinity_pred = [], []
    all_pK_flags = []

    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            res_b, mol_b, tgts = batch_data

            res_b = move_to_device(res_b, device)
            mol_b = move_to_device(mol_b, device)
            tgts  = move_to_device(tgts, device)

            preds = model(res_b, mol_b)

            p_pred = preds[1].detach().cpu()
            p_true = tgts[1].detach().cpu()

            all_p_pred_global.append(p_pred)
            all_p_true_global.append(p_true)

            # Binder-only dilimleme
            res_scope = res_b[2]
            mol_scope = next(x for x in mol_b if isinstance(x, list))
            binder_atoms = mol_scope[0][1]
            pocket_res   = res_scope[0][1]
            binder_len   = binder_atoms * pocket_res

            all_p_pred_binder.append(p_pred[:binder_len])
            all_p_true_binder.append(p_true[:binder_len])

            all_affinity_pred.append(preds[2].detach().cpu())
            all_affinity_true.append(tgts[2].detach().cpu())
            all_pK_flags.append(tgts[3].detach().cpu())

    return {
        'p_true_global': torch.cat(all_p_true_global).numpy(),
        'p_pred_global': torch.cat(all_p_pred_global).numpy(),
        'p_true_binder': torch.cat(all_p_true_binder).numpy(),
        'p_pred_binder': torch.cat(all_p_pred_binder).numpy(),
        'aff_true':      torch.cat(all_affinity_true).numpy(),
        'aff_pred':      torch.cat(all_affinity_pred).numpy(),
        'pk_flags':      torch.cat(all_pK_flags).numpy(),
    }

def find_optimal_threshold(y_true, y_pred):
    """Youden's J istatistiğine göre optimal threshold bulur."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])

def compute_metrics(y_true, y_pred, threshold_opt=None, prefix="", set_name=""):
    """AUROC, AUPRC + sabit 0.5 eşiği + Youden optimal eşiği."""
    auroc = roc_auc_score(y_true, y_pred) if np.sum(y_true) > 0 else 0.5
    auprc = average_precision_score(y_true, y_pred) if np.sum(y_true) > 0 else 0.0

    metrics = {
        f"{set_name}/{prefix}_AUROC": auroc,
        f"{set_name}/{prefix}_AUPRC": auprc,
    }

    # Sabit 0.5 eşiği
    y_bin_05 = (y_pred > 0.5).astype(int)
    metrics[f"{set_name}/{prefix}_F1_05"]        = f1_score(y_true, y_bin_05, zero_division=0)
    metrics[f"{set_name}/{prefix}_Precision_05"] = precision_score(y_true, y_bin_05, zero_division=0)
    metrics[f"{set_name}/{prefix}_Recall_05"]    = recall_score(y_true, y_bin_05, zero_division=0)

    # Youden optimal eşiği
    if threshold_opt is not None:
        y_bin_opt = (y_pred > threshold_opt).astype(int)
        metrics[f"{set_name}/{prefix}_OptThreshold"]  = threshold_opt
        metrics[f"{set_name}/{prefix}_F1_Opt"]        = f1_score(y_true, y_bin_opt, zero_division=0)
        metrics[f"{set_name}/{prefix}_Precision_Opt"] = precision_score(y_true, y_bin_opt, zero_division=0)
        metrics[f"{set_name}/{prefix}_Recall_Opt"]    = recall_score(y_true, y_bin_opt, zero_division=0)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',  required=True,
                        choices=['noesm', 'esm', 'vg', 'contrastive'],
                        help='Model mimarisi: noesm | esm | vg | contrastive')
    parser.add_argument('--ckpt_dir',    required=True)
    parser.add_argument('--valid_data',  required=True)
    parser.add_argument('--test_data',   required=True)
    parser.add_argument('--project',     default='PLANET_THRESHOLD')
    parser.add_argument('--run_name',    required=True)
    parser.add_argument('--dims',        type=int, default=512)
    parser.add_argument('--min_step',    type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wandb.init(project=args.project, name=args.run_name)

    model = PLANET(args.dims, 8, 128, 128, 3, 10, 1, device).to(device)

    ckpts = sorted(
        [f for f in os.listdir(args.ckpt_dir)
         if 'iter-' in f and get_iteration(f) >= args.min_step],
        key=get_iteration
    )
    print(f"Toplam checkpoint: {len(ckpts)} (min_step={args.min_step})")

    v_loader = DataLoader(
        ProLigDataset(args.valid_data, 1, decoy_flag=True, decoy_count=11),
        batch_size=1, collate_fn=lambda x: x[0], num_workers=4
    )
    t_loader = DataLoader(
        ProLigDataset(args.test_data, 1, decoy_flag=True, decoy_count=11),
        batch_size=1, collate_fn=lambda x: x[0], num_workers=4
    )

    for ckpt in ckpts:
        step = get_iteration(ckpt)
        print(f"\n⌛ [{step}] {ckpt} değerlendiriliyor...")

        model.load_state_dict(
            torch.load(os.path.join(args.ckpt_dir, ckpt), map_location=device)
        )

        # ── 1. Valid tahminlerini topla ────────────────────────
        v_data = collect_predictions(model, v_loader, device, 'Valid')

        # ── 2. Valid üzerinden Youden optimal threshold bul ───
        thr_global = find_optimal_threshold(v_data['p_true_global'], v_data['p_pred_global'])
        thr_binder = find_optimal_threshold(v_data['p_true_binder'], v_data['p_pred_binder'])
        print(f"   Youden Threshold — Global: {thr_global:.4f} | BinderOnly: {thr_binder:.4f}")

        # ── 3. Valid metrikleri ────────────────────────────────
        v_metrics = {}
        v_metrics.update(compute_metrics(
            v_data['p_true_global'], v_data['p_pred_global'],
            threshold_opt=thr_global, prefix='Global', set_name='Valid'
        ))
        v_metrics.update(compute_metrics(
            v_data['p_true_binder'], v_data['p_pred_binder'],
            threshold_opt=thr_binder, prefix='BinderOnly', set_name='Valid'
        ))
        mask = v_data['pk_flags'] == 1.0
        v_metrics['Valid/Affinity_Pearson'] = (
            pearsonr(v_data['aff_true'][mask], v_data['aff_pred'][mask])[0]
            if mask.sum() > 1 else 0.0
        )

        # ── 4. Test tahminlerini topla ─────────────────────────
        t_data = collect_predictions(model, t_loader, device, 'Test')

        # ── 5. Test metrikleri — valid threshold ile ───────────
        t_metrics = {}
        t_metrics.update(compute_metrics(
            t_data['p_true_global'], t_data['p_pred_global'],
            threshold_opt=thr_global, prefix='Global', set_name='Test'
        ))
        t_metrics.update(compute_metrics(
            t_data['p_true_binder'], t_data['p_pred_binder'],
            threshold_opt=thr_binder, prefix='BinderOnly', set_name='Test'
        ))
        mask = t_data['pk_flags'] == 1.0
        t_metrics['Test/Affinity_Pearson'] = (
            pearsonr(t_data['aff_true'][mask], t_data['aff_pred'][mask])[0]
            if mask.sum() > 1 else 0.0
        )

        # ── 6. WandB'ye gönder ────────────────────────────────
        wandb.log({**v_metrics, **t_metrics}, step=step, commit=True)

        print(
            f"✅ Valid — Global AUROC: {v_metrics['Valid/Global_AUROC']:.3f} | "
            f"Global F1_Opt: {v_metrics['Valid/Global_F1_Opt']:.3f} | "
            f"BinderOnly F1_Opt: {v_metrics['Valid/BinderOnly_F1_Opt']:.3f}"
        )

    wandb.finish()
    print("\n🎉 Tüm checkpointler tamamlandı.")