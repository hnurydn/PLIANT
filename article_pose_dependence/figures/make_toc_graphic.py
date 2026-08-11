#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_toc_graphic.py
====================
JCIM Table-of-Contents (TOC) graphic. Per ACS TOC instructions (standard
across ACS journals incl. JCIM): max 9.0 cm wide x 3.5 cm tall, 300 dpi
color, Helvetica lettering 6-8 pt, text limited to labeling (not an
abstract, no specific numeric results).

Concept: the paper's central finding in one glance -- as ligand pose
accuracy degrades, a pose-sensitive (Vector-Gating) architecture's apparent
screening performance collapses while a pose-blind architecture (which
never sees ligand coordinates) is completely unaffected. Curve *shapes* are
the true relative trend from pose_perturbation_eval_results.csv
(NoESM_NoUW_baseline vs. ESM_VG, Affinity Pearson r), min-max normalized to
[0,1] for display -- no axis numbers are shown (ACS: no specific results).
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MAKALE_DIR = os.path.dirname(HERE)
FIG_DIR = os.path.join(MAKALE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLOR_BASELINE = "#2a78d6"  # dataviz skill validated palette slot 1
COLOR_VG = "#eb6834"  # slot 2

CM_TO_IN = 1 / 2.54
MAX_W_CM, MAX_H_CM = 9.0, 3.5


def load_curve(csv_path, model_label):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model_label"] == model_label:
                rows.append((float(row["rmsd_bin"]), float(row["Affinity_Pearson"])))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def normalize(y, y_all):
    lo, hi = min(y_all), max(y_all)
    return [(v - lo) / (hi - lo) for v in y]


def main():
    csv_path = os.path.join(MAKALE_DIR, "pose_perturbation_eval_results.csv")
    x_b, y_b = load_curve(csv_path, "NoESM_NoUW_baseline")
    x_v, y_v = load_curve(csv_path, "ESM_VG")

    y_all = y_b + y_v
    y_b_n = normalize(y_b, y_all)
    y_v_n = normalize(y_v, y_all)

    fig_w, fig_h = MAX_W_CM * CM_TO_IN, MAX_H_CM * CM_TO_IN
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x_b, y_b_n, color=COLOR_BASELINE, linewidth=2.2, solid_capstyle="round", zorder=3)
    ax.plot(x_v, y_v_n, color=COLOR_VG, linewidth=2.2, solid_capstyle="round", zorder=3)
    ax.scatter([x_b[-1]], [y_b_n[-1]], color=COLOR_BASELINE, s=18, zorder=4)
    ax.scatter([x_v[-1]], [y_v_n[-1]], color=COLOR_VG, s=18, zorder=4)

    ax.text(x_b[-1] + 0.15, y_b_n[-1], "pose-blind", color=COLOR_BASELINE,
             fontsize=7, fontname="Arial", va="center", ha="left", fontweight="bold")
    ax.text(x_v[-1] + 0.15, y_v_n[-1], "pose-sensitive", color=COLOR_VG,
             fontsize=7, fontname="Arial", va="center", ha="left", fontweight="bold")

    ax.set_xlim(-0.15, 5.6)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xticks([x_b[0], x_b[-1]])
    ax.set_xticklabels(["native\npose", "perturbed\npose"], fontsize=6, fontname="Arial")
    ax.set_yticks([])
    ax.set_ylabel("screening\nperformance", fontsize=6.5, fontname="Arial", labelpad=2)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#52514e")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.14, right=0.78, top=0.97, bottom=0.22)

    out_png = os.path.join(FIG_DIR, "toc_graphic.png")
    out_tiff = os.path.join(FIG_DIR, "toc_graphic.tiff")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_tiff, dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    from PIL import Image
    im = Image.open(out_tiff)
    print(f"yazildi: {out_tiff}  size_px={im.size}  size_cm=({im.size[0]/300*2.54:.2f}, {im.size[1]/300*2.54:.2f})")
    print(f"yazildi: {out_png}")


if __name__ == "__main__":
    main()
