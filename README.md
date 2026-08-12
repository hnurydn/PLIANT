# PLANET — Protein–Ligand Binding Affinity Prediction

This repository contains the training, data-preparation, and evaluation code for
**PLANET**, a graph neural network model for protein–ligand binding affinity
prediction, developed as part of a Master's thesis and extended into a research
article.

The model compares several architectures for protein–ligand affinity prediction:
ESM-2 protein language model embeddings + Uncertainty Weighting (UW) + Vector
Gating (cross-modal geometric attention) + Contrastive Loss (several variants).

> This repo contains **code only**. Trained model weights (`*.safetensors`,
> `PLANET.iter-*`, etc.), large intermediate data files, and logs are excluded
> via `.gitignore` and are hosted separately (see **Data** below).

## Directory Structure

```
.
├── Dataset_preparation/       # Data preparation / preprocessing pipeline (steps 02-20)
├── Model/                     # Legacy/exploratory model development (superseded by Scripts/)
├── Scripts/
│   ├── models/                 # 10 final model variants, one self-contained folder each
│   │   └── <model_name>/       #   train script + model.py + layers/chemutils/datautils/nnutils
│   ├── evaluation/              # Shared DUD-E / LIT-PCBA / CASF benchmark + re-evaluation scripts
│   ├── finetune_crossdocked/    # Separate CrossDocked fine-tuning pathway (not one of the 10 models)
│   └── deprecated/              # Superseded intermediate architecture (kept for record only)
├── article_pose_dependence/   # JCIM article reproducibility package (see below)
├── ligand_lm_extension/       # Ligand language model (ChemBERTa) extension, work in progress
└── *.py, *.sh                 # Top-level driver and Slurm submission scripts
```

See `Scripts/models/README.md` for the exact mapping between each of the 10
reported model variants and the files that implement it.

## Article: Pose-(In)dependence Reproducibility Package

Post-thesis analyses (pose-perturbation experiment, checkpoint robustness,
seed-robustness, DUD-E/LIT-PCBA statistical significance testing, and figures)
supporting the manuscript submitted to the *Journal of Chemical Information
and Modeling* — code and derived data:

📁 **[`article_pose_dependence/`](article_pose_dependence/)**

## Data

Training/evaluation data (LMDB) is large (~34 GB) and is not stored in this
repo; it is hosted as a private dataset on **Hugging Face**:

🤗 **https://huggingface.co/datasets/hnurydn/PLANET-datasets**

| Subfolder | Description | Size |
|-----------|-------------|------|
| `lmdb_planet/` | No protein-LM (no-ESM) — used by `noESM_*` models | ~9 GB |
| `lmdb_planet_v2/` | With ESM-2 embeddings (v2) — used by `ESM_*` / `v2_*` / `v4_*` models | ~25 GB |
| `lmdb_planet_v3/` | v2 + precomputed ChemBERTa ligand embeddings (ligand_lm_extension) | in progress |

To clone (a private-repo login via `huggingface-cli login` is required first):

```bash
git clone https://huggingface.co/datasets/hnurydn/PLANET-datasets
```

or to download a single subfolder:

```bash
huggingface-cli download hnurydn/PLANET-datasets --repo-type dataset \
  --include "lmdb_planet_v2/*" --local-dir datasets/pdbbind2020
```

See [`datasets/pdbbind2020/lmdb_planet/DATASET.md`](datasets/pdbbind2020/lmdb_planet/DATASET.md)
and [`.../lmdb_planet_v2/DATASET.md`](datasets/pdbbind2020/lmdb_planet_v2/DATASET.md) for details.

## Running the code locally

All `.py` scripts read their data-root paths from environment variables
instead of hardcoded paths, so they run on any machine once the data above is
downloaded and the variables below are set:

| Variable | Meaning | Used by |
|---|---|---|
| `PLANET_ROOT` | Root of the PDBbind/PLANET dataset tree (contains `datasets/`, checkpoint folders, etc.) | most scripts in `Dataset_preparation/`, `Scripts/`, `article_pose_dependence/` |
| `BANANA_ROOT` | Install directory of the [BANANA](https://github.com/jacksonwoodruff/BANANA) pose-scoring tool used during decoy preparation | a few `Dataset_preparation/` scripts |
| `ANZU_DATA_ROOT` / `ANZU_HOME` | Local data/home checkout paths, used by a handful of benchmark scripts that were originally run on a secondary workstation | a few `Dataset_preparation/` benchmark scripts |

Example:

```bash
export PLANET_ROOT=/path/to/where/you/put/the/downloaded/dataset
python Scripts/models/ESM_VG/PLANET_train_esm2_vector_gating.py \
    -t $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet_v2/train.lmdb \
    -v $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet_v2/valid.lmdb \
    -te $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet_v2/core.lmdb \
    -d ./checkpoints/ESM_VG
```

**Note on `.sh` files:** the Slurm submission scripts (`*.sh`) are kept as an
exact record of how jobs were originally launched on the TRUBA/ARF HPC
cluster (partition names, account, `sbatch` directives) — they are
cluster-specific and not meant to be portable as-is. The underlying `.py`
scripts they call, however, are portable (see above) and can be run directly
with `python script.py --args ...` on any machine.

## Notes

- Jobs were run on the TRUBA/ARF cluster via **Slurm** (`sbatch *.sh`).
- WandB projects: `PLANET_TRAINING` → `PLANET_COMBINED_RESULT`.
- Conda environment used on the cluster: `pocketgen`.
