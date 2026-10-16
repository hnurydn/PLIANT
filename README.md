# PLIANT — Protein–Ligand Interaction and Affinity Prediction

This repository contains the training, data-preparation and evaluation code for
**PLIANT**, a multi-task graph neural network that predicts protein–ligand
contacts, intra-ligand geometry and binding affinity together.

PLIANT is trained with **non-binding molecules placed inside the binding pocket**
and with **two margin constraints** — a binder must score above a decoy in the
same pocket, and a crystallographic pose must score above a displaced pose of the
same ligand. The aim is to stop a model from reaching a high benchmark score
through ligand-only cues such as molecular size, and to make the predicted
affinity respond to the ligand pose.

> This repository contains **code only**. Trained weights (`PLANET.iter-*`),
> large intermediate data and logs are excluded via `.gitignore` and hosted
> separately (see **Data**).

## Relation to PLANET

PLIANT builds on **PLANET** (Zhang X. et al., *J. Chem. Inf. Model.* 2024;64(7):2205–2220),
and the two are kept clearly separate in this repository:

| From PLANET (upstream) | Added in PLIANT |
|---|---|
| `ProteinEGNN` — geometric message passing over the pocket residue graph | ESM-2 protein language model embeddings as residue features |
| `LigandGAT` — graph attention over the ligand graph | Cross-modal gating that fuses structural and sequence representations per feature dimension |
| `ProLig` / `ProteinLigandAttention` — protein–ligand communication | Vector attention with an explicit interatomic-distance bias (`geo_bias`), which is how the ligand pose enters the model |
| Multi-task objective (contact map + ligand distance matrix + affinity) | In-pocket decoy generation and the two margin losses |

**Module and checkpoint names are deliberately not renamed.** Files such as
`PLANET_model_*.py` and checkpoints named `PLANET.iter-*` follow the upstream
naming so that the code remains traceable to its origin; they are not a claim
that the upstream model is ours.

PLANET is used as a published **reference point**, not as a baseline we tuned:
reported PLANET numbers are either taken from the paper above or obtained by us
by evaluating the published PLANET weights under our own protocol, and the two
cases are labelled separately wherever they appear.

## Model variants

Three variants are compared. They share the training set, protocol, losses,
uncertainty-based task weighting and random seed, and all reported values come
from matched 700k-step checkpoints. The variants differ only in the protein
representation and the pose-aware components:

| Variant | Protein representation | Pose-aware components | Model module | Checkpoint |
|---|---|---|---|---|
| **PLIANT** | BLOSUM62 residue features (`FEATURE_DIM=20`) | none | `PLANET_model_blosum_uw_orj_gercek.py` | `ABL1_P0_700k/PLANET.iter-700000` |
| **PLIANT-ESM** | ESM-2 embeddings (1280-d, PCA-reduced to 512) | none | `PLANET_model_esm2_uw_orj_gercek.py` | `ABL2_P0_700k/PLANET.iter-700000` |
| **PLIANT-ESM-Pose** | ESM-2 embeddings | cross-modal gating + vector attention | `PLANET_model_esm2_vg_orj_ceptemask.py` | `BYER0_700k/PLANET.iter-700000` |

The first two are **pose-blind by construction**: removing the distance-biased
vector attention removes the only path by which ligand coordinates reach the
score, so these variants assign the same affinity to every pose of a ligand.
This is a property of the architecture, not a training artefact, and it is what
the docking-power and pose-margin results measure.

## Directory structure

```
.
├── Dataset_preparation/       # Data preparation / preprocessing pipeline (steps 02-20)
├── Model/                     # Legacy/exploratory model development (superseded by Scripts/)
├── Scripts/
│   ├── models/                 # Earlier model variants from the thesis, one folder each
│   ├── evaluation/             # Shared DUD-E / LIT-PCBA / CASF benchmark scripts
│   ├── finetune_crossdocked/   # Separate CrossDocked fine-tuning pathway
│   └── deprecated/             # Superseded intermediate architecture (kept for record)
├── pliant_hibit2026/          # PLIANT variants: training, in-pocket decoys, evaluation
├── article_pose_dependence/   # Pose-(in)dependence reproducibility package
├── ligand_lm_extension/       # Ligand language model (ChemBERTa) extension, work in progress
└── *.py, *.sh                 # Top-level driver and Slurm submission scripts
```

> **Note.** `Scripts/models/` holds the earlier variants trained *without*
> in-pocket decoys and margins. They share architecture names with the PLIANT
> variants above but are **not the same runs** — the attention-scaling fixes used
> in all three PLIANT arms are not present there. The code for the PLIANT
> variants is in [`pliant_hibit2026/`](pliant_hibit2026/).

## Data

Training/evaluation data (LMDB) is large and is not stored in this repository;
it is hosted as a dataset on **Hugging Face**:

🤗 **https://huggingface.co/datasets/hnurydn/PLANET-datasets**

| Subfolder | Description | Size |
|---|---|---|
| `lmdb_planet/` | No protein-LM (no-ESM) variants | ~9 GB |
| `lmdb_planet_v2/` | With ESM-2 embeddings | ~25 GB |
| `lmdb_planet_v3/` | v2 + precomputed ChemBERTa ligand embeddings | in progress |

The PLIANT variants use separate LMDBs in which every non-binding molecule has
been placed inside the pocket; these are not the LMDBs listed above and will be
documented together with the reproducibility package.

To clone (a login via `huggingface-cli login` is required first):

```bash
git clone https://huggingface.co/datasets/hnurydn/PLANET-datasets
```

or to download a single subfolder:

```bash
huggingface-cli download hnurydn/PLANET-datasets --repo-type dataset \
  --include "lmdb_planet_v2/*" --local-dir datasets/pdbbind2020
```

## Running the code locally

All `.py` scripts read their data-root paths from environment variables instead
of hardcoded paths, so they run on any machine once the data has been downloaded
and the variables below are set:

| Variable | Meaning |
|---|---|
| `PLANET_ROOT` | Root of the dataset tree (contains `datasets/`, checkpoint folders, …) |
| `BANANA_ROOT` | Install directory of the BANANA pose-scoring tool used during decoy preparation |
| `ANZU_DATA_ROOT` / `ANZU_HOME` | Data/home checkout paths for benchmark scripts originally run on a secondary workstation |

Example:

```bash
export PLANET_ROOT=/path/to/the/downloaded/dataset
python Scripts/models/ESM_VG/PLANET_train_esm2_vector_gating.py \
    -t $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet_v2/train.lmdb \
    -v $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet_v2/valid.lmdb \
    -te $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet_v2/core.lmdb \
    -d ./checkpoints/ESM_VG
```

**Note on `.sh` files.** The Slurm submission scripts are kept as an exact record
of how jobs were launched on the TRUBA/ARF cluster (partition names, account,
`sbatch` directives). They are cluster-specific and not portable as-is; the `.py`
scripts they call are portable and can be run directly.

## Notes

- Jobs were run on the TRUBA/ARF cluster via **Slurm** (`sbatch *.sh`).
- Conda environment used on the cluster: `pocketgen`.
- Benchmarks: CASF-2016 (Su M. et al., *J. Chem. Inf. Model.* 2019;59(2):895–913),
  DUD-E (Mysinger M. M. et al., *J. Med. Chem.* 2012;55(14):6582–6594) and
  LIT-PCBA (Tran-Nguyen V.-K. et al., *J. Chem. Inf. Model.* 2020;60(9):4263–4273).
- Training data: PDBbind 2020 general set (Liu Z. et al., *Bioinformatics*
  2015;31(3):405–412). In-pocket decoy practice follows PIGNet
  (Moon S. et al., *Chem. Sci.* 2022;13(13):3661–3673).
