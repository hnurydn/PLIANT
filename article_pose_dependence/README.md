# Reproducibility Package — "Pose-(In)dependence" Article

This folder contains the code, derived data, and figures supporting the
journal manuscript built on top of the PLANET thesis project (submitted to
*Journal of Chemical Information and Modeling*, JCIM). It is a sub-project of
the main [PLANET-thesis](../) repository and reuses its model definitions,
training pipeline, and datasets (see top-level [README](../README.md) and
the [Hugging Face dataset](https://huggingface.co/datasets/hnurydn/PLANET-datasets)).

Everything here maps directly onto the Methods subsections of the manuscript:

| Folder | Manuscript section | What it does |
|---|---|---|
| [`pose_perturbation/`](pose_perturbation/) | §2.5 Pose-Perturbation Experiment | RDKit rigid-body pose perturbation calibrated to a target RMSD via binary search, and GPU evaluation of 3 model variants across 5 RMSD bins |
| [`checkpoint_analysis/`](checkpoint_analysis/) | §2.6 Checkpoint Robustness Analysis | Reconstructs per-checkpoint test-metric trajectories for all 7 model variants directly from WandB training logs (no extra GPU compute) |
| [`seed_robustness/`](seed_robustness/) | §2.7 Seed-Robustness Protocol | Seeded re-training scripts (3 seeds) for the baseline, ESM_VG, and v4_Projection variants, and their Slurm launchers |
| [`statistical_analysis/`](statistical_analysis/) | §2.8 Statistical Significance Testing | Per-target one-sided Wilcoxon signed-rank analysis on DUD-E (101 targets) and LIT-PCBA (15 targets), for both "centered" and "original" (non-recentered) ligand-pose preprocessing |
| [`figures/`](figures/) | Figure 1, Figure S1 | Generates the pose-dependence and checkpoint-plateau figures from the CSVs above |

## Key finding these scripts establish

Only model variants that pass ligand coordinates into their forward
computation (the Vector-Gating family) degrade under pose perturbation and
are inflated by ligand-pose "centering" preprocessing on DUD-E/LIT-PCBA.
Architecturally pose-blind variants (baseline, ESM_NoUW, ESM_UW) are
unaffected by both manipulations — a built-in negative control confirmed
through three independent lines of evidence (perturbation, checkpoint
trajectory, and real-benchmark centering artifact).

## Requirements

Same conda environment as the main repo (`pocketgen`), plus:
- `rdkit` (pose perturbation / RMSD calibration)
- `wandb` (checkpoint-trajectory reconstruction; read access to the
  `h-nurydn-hacettepe-university/PLANET_TRAINING` and
  `PLANET_COMBINED_RESULT` projects)
- `scipy`, `pandas`, `matplotlib` (statistics and figures)

## Data not included here

- Raw benchmark data (PDBbind, CASF-2016, DUD-E, LIT-PCBA) are third-party
  and publicly available from their original sources; not redistributed.
- The base LMDB training/test datasets (`lmdb_planet/`, `lmdb_planet_v2/`,
  ~34 GB) live in the [Hugging Face dataset
  repo](https://huggingface.co/datasets/hnurydn/PLANET-datasets), not here.
- Trained model checkpoints (`.safetensors`) used to produce
  `pose_perturbation_eval_results.csv` and the WandB-derived curves are
  referenced by run ID in `checkpoint_analysis/fetch_checkpoint_curves_from_wandb.py`
  and are not redistributed in this repo (see Data Availability Statement
  in the manuscript for archival status).

## Notes on script provenance

`pose_perturbation/` scripts operate on two different LMDB datasets
depending on model family: `lmdb_planet/core.lmdb` (no-ESM, 20-dim residue
features) for the baseline, and `lmdb_planet_v2/core.lmdb` (512-dim ESM-2
features) for ESM_VG / v4_Projection — each perturbed LMDB copy carries an
auto-copied sibling `_meta.json` required by `ProLigDataset`.

`seed_robustness/` scripts are seeded, `max_step`-patched rewrites of the
production training scripts under the main repo's `Scripts/` folder (not
`Model/` — the latter is stale/unused), with the WandB destination project
changed to `PLANET_SEED_ROBUSTNESS` to keep these runs separate from the
main `PLANET_TRAINING` history used for checkpoint analysis.
