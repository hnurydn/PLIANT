#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
================
Makale icin iki yayin-kalitesinde figur uretir:
  1. figure_pose_dependence.(pdf|png) -- ANA FIGUR: RMSD vs performans (2 panel),
     3 model (baseline/ESM_VG/v4_Projection), pose_perturbation_eval_results.csv'den.
  2. figure_S_checkpoint_plateau.(pdf|png) -- SUPPLEMENTARY: egitim adimina karsi
     Test/Affinity_Pearson (7 model, kucuk-coklu), 1.9M checkpoint'i isaretli,
     checkpoint_curves_from_wandb.csv'den.

Renkler: dataviz skill'in validate edilmis kategorik paletinin ilk 3 slotu (tum-ciftler
CVD-guvenli, hem acik hem koyu modda): mavi/turuncu/aqua.
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
MAKALE_DIR = os.path.dirname(HERE)
FIG_DIR = os.path.join(MAKALE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── dataviz skill palette (slot 1-3, tum-ciftler dogrulanmis) ─────────────────
COLOR_BASELINE = "#2a78d6"   # slot 1: blue
COLOR_ESM_VG   = "#eb6834"   # slot 2: orange
COLOR_V4PROJ   = "#1baf7a"   # slot 3: aqua

MODEL_STYLE = {
    "NoESM_NoUW_baseline": {"color": COLOR_BASELINE, "marker": "o", "ls": "-",  "label": "Baseline (no pose input)"},
    "ESM_VG":               {"color": COLOR_ESM_VG,   "marker": "s", "ls": "--", "label": "ESM_VG"},
    "v4_Projection":         {"color": COLOR_V4PROJ,   "marker": "^", "ls": "-.", "label": "v4_Projection"},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e5e4df",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 300,
})


# ═══════════════════════════ Figure 1: ana pose-dependence figuru ══════════════

def load_pose_perturbation_results(path):
    data = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["model_label"]].append(row)
    for model in data:
        data[model].sort(key=lambda r: float(r["rmsd_bin"]))
    return data


def make_figure_pose_dependence(csv_path):
    data = load_pose_perturbation_results(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)

    panels = [
        ("BinderOnly_AUROC", "Binder-only contact AUROC", axes[0], (0.80, 1.0)),
        ("Affinity_Pearson", "Affinity Pearson r", axes[1], (0.55, 0.85)),
    ]

    for metric_key, ylabel, ax, ylim in panels:
        for model_key, style in MODEL_STYLE.items():
            rows = data[model_key]
            x = [float(r["rmsd_bin"]) for r in rows]
            y = [float(r[metric_key]) for r in rows]
            ax.plot(
                x, y,
                color=style["color"], marker=style["marker"], linestyle=style["ls"],
                linewidth=2, markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                label=style["label"], zorder=3,
            )
        ax.set_xlabel("Ligand pose RMSD from native (Å)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_ylim(*ylim)
        ax.margins(x=0.04)

    # Tek ortak lejant (alt panel), iki eksen arasinda tekrar etmesin
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.08), fontsize=9,
    )

    axes[0].text(-0.14, 1.05, "A", transform=axes[0].transAxes,
                 fontsize=13, fontweight="bold", va="top")
    axes[1].text(-0.14, 1.05, "B", transform=axes[1].transAxes,
                 fontsize=13, fontweight="bold", va="top")

    out_pdf = os.path.join(FIG_DIR, "figure_pose_dependence.pdf")
    out_png = os.path.join(FIG_DIR, "figure_pose_dependence.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", out_pdf)
    print("yazildi:", out_png)


# ═══════════════════════════ Figure S: checkpoint plato supplementary ══════════

MODEL_ORDER_S = [
    "NoESM_NoUW_baseline", "ESM_NoUW", "ESM_UW", "NoESM_UW",
    "ESM_VG", "ESM_VG_Contrastive", "v4_Projection",
]
MODEL_LABEL_S = {
    "NoESM_NoUW_baseline": "Baseline",
    "ESM_NoUW": "ESM_NoUW",
    "ESM_UW": "ESM_UW",
    "NoESM_UW": "NoESM_UW",
    "ESM_VG": "ESM_VG",
    "ESM_VG_Contrastive": "ESM_VG_Contrastive",
    "v4_Projection": "v4_Projection",
}
CHOSEN_STEP = 1_900_000


def load_checkpoint_curves(path):
    data = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["model"]].append(row)
    for model in data:
        data[model].sort(key=lambda r: float(r["step"]))
    return data


def make_figure_checkpoint_plateau(csv_path):
    data = load_checkpoint_curves(csv_path)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), constrained_layout=True, sharex=True)
    axes_flat = axes.flatten()

    line_color = "#4a3aa7"  # slot 7: violet -- ana figurden ayri, tek-seri renk

    for i, model_key in enumerate(MODEL_ORDER_S):
        ax = axes_flat[i]
        rows = data[model_key]
        x = [float(r["step"]) / 1e6 for r in rows]
        y = [float(r["Test_Affinity_Pearson"]) for r in rows]

        ax.plot(x, y, color=line_color, linewidth=1.2, alpha=0.85, zorder=2)
        ax.axvline(CHOSEN_STEP / 1e6, color="#e34948", linewidth=1.2,
                   linestyle="--", zorder=1, label="Chosen checkpoint (1.9M)")

        chosen_rows = [r for r in rows if abs(float(r["step"]) - CHOSEN_STEP) < 1]
        if chosen_rows:
            cy = float(chosen_rows[0]["Test_Affinity_Pearson"])
            ax.scatter([CHOSEN_STEP / 1e6], [cy], color="#e34948", s=28, zorder=3)

        ax.set_title(MODEL_LABEL_S[model_key], fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=8)

    # kullanilmayan 8. panel -> lejant icin
    axes_flat[7].axis("off")
    legend_handles = [
        plt.Line2D([0], [0], color=line_color, linewidth=1.5, label="Test Affinity Pearson (per checkpoint)"),
        plt.Line2D([0], [0], color="#e34948", linewidth=1.5, linestyle="--", marker="o",
                   markersize=5, label="Chosen checkpoint (iter. 1,900,000)"),
    ]
    axes_flat[7].legend(handles=legend_handles, loc="center", fontsize=9, frameon=False)

    fig.supxlabel("Training step (millions)", fontsize=10.5)
    fig.supylabel("Test set Affinity Pearson r", fontsize=10.5)

    out_pdf = os.path.join(FIG_DIR, "figure_S_checkpoint_plateau.pdf")
    out_png = os.path.join(FIG_DIR, "figure_S_checkpoint_plateau.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("yazildi:", out_pdf)
    print("yazildi:", out_png)


if __name__ == "__main__":
    make_figure_pose_dependence(os.path.join(MAKALE_DIR, "pose_perturbation_eval_results.csv"))
    make_figure_checkpoint_plateau(os.path.join(MAKALE_DIR, "checkpoint_curves_from_wandb.csv"))
