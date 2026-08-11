#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_perturbation_eval.py
===========================
Bir PLANET modelini (tek checkpoint) pose_perturbation.py'nin ürettiği RMSD bin'li
CASF-2016 core-set LMDB'lerine karşı değerlendirir. Amaç: Affinity Pearson / Contact
AUROC'un ligand poz RMSD'sinin fonksiyonu olarak nasıl bozulduğunu ölçmek
(bkz. makale/pose_perturbation_teknik_tasarim.md).

Tasarım kararları:
  - TEK checkpoint (iter-1900000) — eval_unified.py'deki gibi checkpoint sweep YOK.
  - Youden threshold, modelin ORİJİNAL (bozulmamış) valid.lmdb'sinden BİR KEZ hesaplanır
    ve tüm RMSD bin'lerine sabit uygulanır. Amaç: eşiği her bin'de yeniden optimize etmek
    "RMSD'nin etkisini" eşik-ayarlamasıyla maskeleyebilir — bu deney için istenmeyen bir
    karıştırıcı (confounder) olurdu.
  - Çıktı WandB'ye değil, doğrudan CSV'ye yazılır (offline/hızlı, auth gerektirmez).

Kullanım (tek model, tüm RMSD bin'leri):
    python pose_perturbation_eval.py \
        --model_type vg \
        --model_label ESM_VG \
        --ckpt_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_UW_VECTOR_GATING_1:11/PLANET.iter-1900000 \
        --valid_data /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2/valid.lmdb \
        --test_data_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_output_esm \
        --rmsd_bins 0,1,2,3,4 \
        --dims 512 \
        --out_csv /arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_eval_results.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score, roc_curve,
                              mean_absolute_error)
from torch.utils.data import DataLoader


# ───────────────────────────── Model dispatch ─────────────────────────────
# eval_unified.py'deki dinamik import deseni genişletildi: contrastive_v4 eklendi
# (v4_Projection modeli, projection-head'li kontrastif varyant).

_MODEL_DISPATCH = {
    "noesm":         ("PLANET_datautils",            "PLANET_model"),
    "esm":           ("PLANET_datautils_esm2",       "PLANET_model_esm2"),
    "vg":            ("PLANET_datautils_esm2_vector", "PLANET_model_esm2_vector_gating"),
    "contrastive":   ("PLANET_datautils_esm2_vector", "PLANET_model_esm2_vector_gating_contrastive_loss"),
    "contrastive_v4": ("PLANET_datautils_esm2_vector", "PLANET_model_esm2_vector_gating_contrastive_loss_v4"),
}


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        res = [move_to_device(t, device) for t in data]
        return tuple(res) if isinstance(data, tuple) else res
    return data


def collect_predictions(model, loader, device):
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
            tgts = move_to_device(tgts, device)

            preds = model(res_b, mol_b)
            p_pred = preds[1].detach().cpu()
            p_true = tgts[1].detach().cpu()
            all_p_pred_global.append(p_pred)
            all_p_true_global.append(p_true)

            res_scope = res_b[2]
            mol_scope = next(x for x in mol_b if isinstance(x, list))
            binder_atoms = mol_scope[0][1]
            pocket_res = res_scope[0][1]
            binder_len = binder_atoms * pocket_res
            all_p_pred_binder.append(p_pred[:binder_len])
            all_p_true_binder.append(p_true[:binder_len])

            all_affinity_pred.append(preds[2].detach().cpu())
            all_affinity_true.append(tgts[2].detach().cpu())
            all_pK_flags.append(tgts[3].detach().cpu())

    return {
        "p_true_global": torch.cat(all_p_true_global).numpy(),
        "p_pred_global": torch.cat(all_p_pred_global).numpy(),
        "p_true_binder": torch.cat(all_p_true_binder).numpy(),
        "p_pred_binder": torch.cat(all_p_pred_binder).numpy(),
        "aff_true": torch.cat(all_affinity_true).numpy(),
        "aff_pred": torch.cat(all_affinity_pred).numpy(),
        "pk_flags": torch.cat(all_pK_flags).numpy(),
    }


def find_optimal_threshold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_metrics(y_true, y_pred, threshold):
    auroc = roc_auc_score(y_true, y_pred) if np.sum(y_true) > 0 else 0.5
    auprc = average_precision_score(y_true, y_pred) if np.sum(y_true) > 0 else 0.0
    y_bin = (y_pred > threshold).astype(int)
    return {
        "AUROC": auroc, "AUPRC": auprc,
        "F1": f1_score(y_true, y_bin, zero_division=0),
        "Precision": precision_score(y_true, y_bin, zero_division=0),
        "Recall": recall_score(y_true, y_bin, zero_division=0),
    }


def evaluate_on_lmdb(model, lmdb_path, device, threshold_global, threshold_binder,
                      decoy_count=11, num_workers=4):
    from importlib import import_module
    # ProLigDataset zaten global scope'ta import edilmiş olacak (main() içinde) —
    # burada sadece DataLoader kuruyoruz.
    global ProLigDataset
    loader = DataLoader(
        ProLigDataset(lmdb_path, 1, decoy_flag=True, decoy_count=decoy_count),
        batch_size=1, collate_fn=lambda x: x[0], num_workers=num_workers,
    )
    data = collect_predictions(model, loader, device)

    metrics = {}
    g = compute_metrics(data["p_true_global"], data["p_pred_global"], threshold_global)
    b = compute_metrics(data["p_true_binder"], data["p_pred_binder"], threshold_binder)
    for k, v in g.items():
        metrics[f"Global_{k}"] = v
    for k, v in b.items():
        metrics[f"BinderOnly_{k}"] = v

    mask = data["pk_flags"] == 1.0
    if mask.sum() > 1:
        metrics["Affinity_Pearson"] = pearsonr(data["aff_true"][mask], data["aff_pred"][mask])[0]
        metrics["Affinity_MAE"] = mean_absolute_error(data["aff_true"][mask], data["aff_pred"][mask])
    else:
        metrics["Affinity_Pearson"] = float("nan")
        metrics["Affinity_MAE"] = float("nan")
    metrics["n_complexes"] = int(mask.sum())
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_type", required=True, choices=list(_MODEL_DISPATCH.keys()))
    ap.add_argument("--model_label", required=True, help="CSV'de görünecek okunabilir model adı (ör. ESM_VG)")
    ap.add_argument("--ckpt_path", required=True, help="TEK bir checkpoint dosyasının tam yolu")
    ap.add_argument("--valid_data", required=True,
                     help="Modelin ORİJİNAL (bozulmamış) valid.lmdb yolu — threshold BİR KEZ buradan hesaplanır")
    ap.add_argument("--test_data_dir", required=True,
                     help="pose_perturbation.py çıktı dizini (core_rmsd{R}_rigid.lmdb dosyalarını içerir)")
    ap.add_argument("--rmsd_bins", default="0,1,2,3,4")
    ap.add_argument("--mode", default="rigid", help="pose_perturbation.py'deki --mode ile eşleşmeli (dosya adı için)")
    ap.add_argument("--dims", type=int, default=512)
    ap.add_argument("--scripts_path", required=True, help="PLANET_model*.py / PLANET_datautils*.py'nin bulunduğu dizin")
    ap.add_argument("--out_csv", required=True, help="Sonuçların EKLENECEĞİ (append) CSV dosyası")
    args = ap.parse_args()

    sys.path.insert(0, args.scripts_path)

    datautils_mod, model_mod = _MODEL_DISPATCH[args.model_type]
    global ProLigDataset
    ProLigDataset = getattr(__import__(datautils_mod, fromlist=["ProLigDataset"]), "ProLigDataset")
    PLANET = getattr(__import__(model_mod, fromlist=["PLANET"]), "PLANET")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pose_perturbation_eval] model_type={args.model_type} label={args.model_label} device={device}")

    model = PLANET(args.dims, 8, 128, 128, 3, 10, 1, device).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"[pose_perturbation_eval] checkpoint yüklendi: {args.ckpt_path}")

    # ── 1. Threshold'u SADECE native valid setinden, BİR KEZ hesapla ──
    v_loader = DataLoader(
        ProLigDataset(args.valid_data, 1, decoy_flag=True, decoy_count=11),
        batch_size=1, collate_fn=lambda x: x[0], num_workers=4,
    )
    v_data = collect_predictions(model, v_loader, device)
    thr_global = find_optimal_threshold(v_data["p_true_global"], v_data["p_pred_global"])
    thr_binder = find_optimal_threshold(v_data["p_true_binder"], v_data["p_pred_binder"])
    print(f"[pose_perturbation_eval] native valid threshold — Global: {thr_global:.4f} | BinderOnly: {thr_binder:.4f}")

    # ── 2. Her RMSD bin'inde bu SABİT threshold ile test et ──
    rmsd_bins = [float(x) for x in args.rmsd_bins.split(",")]
    rows = []
    for rmsd in rmsd_bins:
        lmdb_path = os.path.join(args.test_data_dir, f"core_rmsd{rmsd:g}_{args.mode}.lmdb")
        if not os.path.exists(lmdb_path):
            print(f"  [!] bulunamadı, atlanıyor: {lmdb_path}")
            continue
        print(f"  -> RMSD={rmsd} değerlendiriliyor: {lmdb_path}")
        metrics = evaluate_on_lmdb(model, lmdb_path, device, thr_global, thr_binder)
        row = {
            "model_label": args.model_label, "model_type": args.model_type,
            "rmsd_bin": rmsd, "ckpt": os.path.basename(args.ckpt_path),
            "valid_threshold_global": thr_global, "valid_threshold_binder": thr_binder,
            **metrics,
        }
        rows.append(row)
        print(f"     Global_AUROC={metrics['Global_AUROC']:.4f}  "
              f"BinderOnly_AUROC={metrics['BinderOnly_AUROC']:.4f}  "
              f"Affinity_Pearson={metrics['Affinity_Pearson']:.4f}  "
              f"Affinity_MAE={metrics['Affinity_MAE']:.4f}")

    # ── 3. CSV'ye ekle (dosya yoksa header ile oluştur) ──
    file_exists = os.path.exists(args.out_csv)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[pose_perturbation_eval] {len(rows)} satır eklendi: {args.out_csv}")


if __name__ == "__main__":
    main()
