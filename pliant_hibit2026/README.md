# PLIANT — HIBIT 2026 reproducibility package

This folder holds the code behind the three PLIANT variants and every number
reported in the HIBIT 2026 abstract: the training modules, the in-pocket decoy
pipeline and the evaluation scripts, together with the Slurm launchers that were
used, so that each reported value can be traced to the script that produced it.

Files were renamed from their cluster filenames to names that match the variant
names used in the abstract. [`PROVENANCE.json`](PROVENANCE.json) records, for
every file, its original filename on the cluster and the md5 of the copy taken.

## The three variants

All three share the training set, the in-pocket decoy protocol, the losses,
uncertainty-based task weighting and random seed (42), and all reported values
come from matched 700k-step checkpoints. They differ only in the protein
representation and the pose-aware components.

| Variant | Model module (training) | Model module (evaluation) | Checkpoint |
|---|---|---|---|
| **PLIANT** | [`train/pliant_model.py`](train/pliant_model.py) | same | `ABL1_P0_700k/PLANET.iter-700000` |
| **PLIANT-ESM** | [`train/pliant_esm_model.py`](train/pliant_esm_model.py) | same | `ABL2_P0_700k/PLANET.iter-700000` |
| **PLIANT-ESM-Pose** | [`train/pliant_esm_pose_train_model.py`](train/pliant_esm_pose_train_model.py) | [`train/pliant_esm_pose_model.py`](train/pliant_esm_pose_model.py) | `BYER0_700k/PLANET.iter-700000` |

PLIANT uses BLOSUM62 residue features (`FEATURE_DIM=20`, run with
`PLANET_BLOSUM=1`); PLIANT-ESM replaces them with ESM-2 embeddings; PLIANT-ESM-Pose
adds cross-modal gating and distance-biased vector attention on top.

For PLIANT-ESM-Pose the training and evaluation modules differ: the two files have
an identical forward pass and differ only in `compute_loss`, so the checkpoint is
loaded for evaluation with the second module. Both are included.

| Supporting file | Role |
|---|---|
| [`train/layers_pliant_esm.py`](train/layers_pliant_esm.py) | layers for PLIANT and PLIANT-ESM |
| [`train/layers_pliant_esm_pose.py`](train/layers_pliant_esm_pose.py) | layers for PLIANT-ESM-Pose, including the gating and the distance-biased attention |
| [`train/train_pliant.py`](train/train_pliant.py) | training driver for PLIANT and PLIANT-ESM |
| [`train/train_pliant_esm_pose.py`](train/train_pliant_esm_pose.py) | training driver for PLIANT-ESM-Pose |
| [`train/margin_losses.py`](train/margin_losses.py) | both margins: `pose_margin_loss` and `molecule_margin_loss` |
| [`train/nnutils.py`](train/nnutils.py), [`train/chemutils_esm2_vector.py`](train/chemutils_esm2_vector.py) | tensor helpers and pocket/ligand featurisation, from the upstream PLANET codebase |
| [`train/slurm_train_continue.sh`](train/slurm_train_continue.sh) | continuation launcher; `TIP=blosum\|esmuw\|byer` selects the variant |

**The first two variants are pose-blind by construction.** The only path from
ligand coordinates to the predicted score is the distance bias inside vector
attention (`ProteinLigandAttention.geo_bias`). Without it the model cannot
distinguish two poses of the same ligand — an architectural property, and what the
docking-power and pose-margin results measure.

## In-pocket decoys and the two margins

Every non-binding molecule is moved into the pocket instead of being left far
outside it:

| Step | File |
|---|---|
| Placement core — 30 random orientations, clash penalty from van der Waals radii, contact reward, then local relaxation with AutoDock Vina | [`decoys/inpocket_placement.py`](decoys/inpocket_placement.py) |
| Driver that walks a split and writes the placed poses | [`decoys/place_decoys.py`](decoys/place_decoys.py) |
| Builds the LMDB consumed by training | [`decoys/build_inpocket_lmdb.py`](decoys/build_inpocket_lmdb.py) |
| Launchers (the split is passed as `SPLIT=` at submit time) | [`decoys/slurm_place_decoys.sh`](decoys/slurm_place_decoys.sh), [`decoys/slurm_place_decoys_valid.sh`](decoys/slurm_place_decoys_valid.sh), [`decoys/slurm_build_lmdb.sh`](decoys/slurm_build_lmdb.sh), [`decoys/slurm_build_lmdb_valid.sh`](decoys/slurm_build_lmdb_valid.sh) |

The two margins are implemented in [`train/margin_losses.py`](train/margin_losses.py)
and applied from the training drivers: a binder must score at least 1 pK above a
decoy in the same pocket (`molecule_margin_loss`), and a crystallographic pose at
least 1 pK above the same ligand displaced by 1–3 Å (`pose_margin_loss`).

## Which script produced which reported number

| Reported value | Produced by | Launcher |
|---|---|---|
| CASF-2016 Pearson *r* (0.824 / 0.809 / 0.844) and MAE (1.004 / 0.973 / 0.920) | the training run's own evaluation on the 271-complex core LMDB | [`train/train_pliant.py`](train/train_pliant.py), [`train/train_pliant_esm_pose.py`](train/train_pliant_esm_pose.py) |
| CASF-2016 contact-map AUROC (0.866 / 0.880 / 0.958) | the same training-time evaluation (`Test/Contact_AUROC`) | as above |
| — independent check of the above, binder-only, 284 targets | [`eval/eval_contact_auroc.py`](eval/eval_contact_auroc.py) | [`eval/slurm_contact_auroc.sh`](eval/slurm_contact_auroc.sh) |
| CASF-2016 docking power (24.4 / 24.4 / 71.5 %) | [`eval/eval_docking_power.py`](eval/eval_docking_power.py) | [`eval/slurm_docking_power_baseline.sh`](eval/slurm_docking_power_baseline.sh), [`eval/slurm_docking_power_esm.sh`](eval/slurm_docking_power_esm.sh), [`eval/slurm_docking_power_pose.sh`](eval/slurm_docking_power_pose.sh) |
| DUD-E, 101 targets, centred (0.544 / 0.640 / 0.545) | [`eval/screen_dude.py`](eval/screen_dude.py) | [`eval/slurm_screen_dude.sh`](eval/slurm_screen_dude.sh) with `MERKEZ=1` |
| LIT-PCBA, 15 targets, centred (0.499 / 0.476 / 0.548) | [`eval/screen_litpcba.py`](eval/screen_litpcba.py) | [`eval/slurm_screen_litpcba.sh`](eval/slurm_screen_litpcba.sh) with `MERKEZ=1` |

Two details needed to read these scripts correctly:

- **`MERKEZ=1` is the centred protocol.** It passes `--center_ligands`, which moves
  each ligand's centre onto the pocket centre with no rotation and no clash
  correction. `MERKEZ=0` is the uncentred protocol used elsewhere in the project and
  is *not* what the abstract reports.
- **Docking power ranks poses by the predicted affinity.** `eval_docking_power.py`
  also computes contact-map agreement channels; the affinity channel is the one
  reported. Ties are broken at random and averaged over 200 repeats (`N_TIE=200`,
  `RMSD_CUT=2.0`), which is necessary because a pose-blind variant gives every pose
  the same score and would otherwise score far above chance by always taking the
  first pose in the file.

## Differences from the copies that ran on the cluster

The code is otherwise unchanged, but two files were reduced to the scope of this
abstract, and one label was corrected:

- `eval/eval_docking_power.py` — the model registry listed about forty further arms
  of the project with their cluster paths; only the three PLIANT entries are kept.
- `train/slurm_train_continue.sh` — the branch for the contrastive arm was removed;
  `TIP=blosum|esmuw|byer` remain.
- In the cluster registry, `ABL2_P0_700k` (PLIANT-ESM) was flagged pose-aware. That
  is a labelling error: the variant has no distance-biased vector attention and its
  tie ratio in this test is 1.000. The flag selects a printed label and a JSON field
  and does not affect the computation; it is set to `False` here to match the
  measurement.

## What is not here

- **Trained weights.** `PLANET.iter-700000` checkpoints are not in git.
- **Data.** The LMDBs with in-pocket decoys are large and hosted separately; they are
  not the thesis LMDBs listed in the root README.
- **Analyses not reported in the abstract.** Pocket-swap specificity, docked-pose
  screening and the contrastive variants were measured but are not part of the
  abstract, so their scripts are not included.

## Notes

- Class names inside the modules keep the upstream `PLANET` naming; see "Relation to
  PLANET" in the repository root README.
- The `.sh` files record how the jobs were submitted on the TRUBA/ARF cluster
  (partition, account, array layout) and contain cluster paths. They are not portable
  as-is; the `.py` scripts they call are the reusable part.
- Some runs started on a local GPU workstation and were continued on the cluster from
  a saved checkpoint; `slurm_train_continue.sh` is the continuation launcher, and it
  neutralises the training scripts' hardcoded `sys.path` entries with `sed` at submit
  time.
