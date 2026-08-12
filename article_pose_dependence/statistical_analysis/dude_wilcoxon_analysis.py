#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dude_wilcoxon_analysis.py
===========================
DUD-E (101 hedef, 10 model) icin per-target istatistiksel anlamlilik analizi.
Tezdeki Tablo 4.8'in (LIT-PCBA, 15 hedef) AYNI metodolojisi -- tek-orneklem
Wilcoxon isaretli-sira testi (AUROC vs rastgele 0.5) + hedefler uzerinde
eslestirilmis ikili Wilcoxon testi -- burada DUD-E'nin 101 hedefine uygulanmis.

Neden onemli: tez metninde (s.71) DUD-E icin bu analiz YAPILMAMISTI, sadece
"istatistiksel anlamlilik acisindan ayrica degerlendirilmelidir" notu dusulmustu.
Makale icin bu bosluk dolduruldu -- ve DUD-E'nin cok daha yuksek istatistiksel
gucu (101 vs 15 hedef) LIT-PCBA'dan FARKLI, daha zengin bir tablo ortaya cikardi
(7/10 model anlamli, LIT-PCBA'da sadece 1/10).

Veri kaynagi: $PLANET_ROOT/dude_results_centered/
(indirilmis kopya: makale/dude_results_centered/), her model icin 101 satir
(Target, ROC_AUC, PR_AUC, n_actives, n_decoys, EF_0.5/1/2/5%).
"""

import csv
import itertools
import os

import numpy as np
from scipy.stats import wilcoxon
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dude_results_centered")

FILE_TO_MODEL = {
    "noESM_noUW_centered.csv": "NoESM_NoUW (baseline)",
    "noESM_UW_centered.csv": "NoESM_UW",
    "ESM_NoUW_centered.csv": "ESM_NoUW",
    "ESM_UW_centered.csv": "ESM_UW",
    "ESM_VG_centered.csv": "ESM_VG",
    "ESM_VG_Contrastive_centered.csv": "ESM_VG_Contrastive",
    "ESM_VG_Contrastive_v2_Mean_m0_centered.csv": "v2_Mean_m0",
    "ESM_VG_Contrastive_v2_Softmax_m0_centered.csv": "v2_Softmax_m0",
    "ESM_VG_Contrastive_v2_Softmax_m01_centered.csv": "v2_Softmax_m01",
    "ESM_VG_Contrastive_v4_Projection_centered.csv": "v4_Projection",
}


def load_data():
    data = {}
    for fname, model in FILE_TO_MODEL.items():
        path = os.path.join(DIR, fname)
        d = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d[row["Target"]] = (float(row["ROC_AUC"]), float(row["EF_1.0%"]))
        data[model] = d
    return data


def one_sample_tests(data):
    common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    results = []
    for model, d in data.items():
        auroc = np.array([d[t][0] for t in common])
        ef1 = np.array([d[t][1] for t in common])
        diff = auroc - 0.5
        nonzero = diff[diff != 0]
        stat, p = wilcoxon(nonzero)
        n_above = int((auroc > 0.5).sum())
        results.append({
            "model": model, "n_targets": len(common),
            "AUROC": auroc.mean(), "EF1%": ef1.mean(),
            "n_above_0.5": n_above, "p_value": p, "significant": p < 0.05,
        })
    results.sort(key=lambda r: -r["AUROC"])
    return results, common


def pairwise_tests(data, common):
    arrays = {m: np.array([d[t][0] for t in common]) for m, d in data.items()}
    models = list(arrays.keys())
    rows = []
    for m1, m2 in itertools.combinations(models, 2):
        diff = arrays[m1] - arrays[m2]
        nonzero = diff[diff != 0]
        if len(nonzero) < 5:
            continue
        stat, p = wilcoxon(nonzero)
        rows.append({
            "model_1": m1, "model_2": m2, "mean_diff": diff.mean(),
            "p_value": p, "significant": p < 0.05,
            "better": m1 if diff.mean() > 0 else m2,
        })
    return rows


if __name__ == "__main__":
    data = load_data()
    one_sample, common = one_sample_tests(data)
    pairwise = pairwise_tests(data, common)

    print(f"=== Tek-örneklem Wilcoxon (AUROC vs 0.5), {len(common)} hedef ===")
    for r in one_sample:
        print(f"{r['model']:22s} AUROC={r['AUROC']:.3f} EF1%={r['EF1%']:.3f} "
              f">0.5={r['n_above_0.5']}/101 p={r['p_value']:.4g} "
              f"{'Anlamlı' if r['significant'] else 'Değil'}")

    sig_pairwise = [r for r in pairwise if r["significant"]]
    print(f"\n=== İkili Wilcoxon: {len(sig_pairwise)}/{len(pairwise)} çift anlamlı (p<0.05) ===")
    for r in sorted(sig_pairwise, key=lambda x: x["p_value"]):
        print(f"{r['model_1']:22s} vs {r['model_2']:22s} "
              f"diff={r['mean_diff']:+.4f} p={r['p_value']:.4g} (daha iyi: {r['better']})")
