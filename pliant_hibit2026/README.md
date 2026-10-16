# PLIANT — HIBIT 2026 reproducibility package

This folder holds the code behind the three PLIANT variants and every number
reported in the HIBIT 2026 abstract. It is a curated subset: the training
modules, the in-pocket decoy pipeline and the evaluation scripts, together with
the Slurm launchers that were actually used, so that each reported value can be
traced to the script that produced it.

## The three variants

All three share the training set, the in-pocket decoy protocol, the losses,
uncertainty-based task weighting and random seed (42), and all reported values
come from matched 700k-step checkpoints. They differ only in the protein
representation and the pose-aware components.

| Variant | Model module | Checkpoint | Notes |
|---|---|---|---|
| **PLIANT** | [`train/PLANET_model_blosum_uw_orj_gercek.py`](train/PLANET_model_blosum_uw_orj_gercek.py) | `ABL1_P0_700k/PLANET.iter-700000` | BLOSUM62 residue features, `FEATURE_DIM=20`; run with `PLANET_BLOSUM=1` |
| **PLIANT-ESM** | [`train/PLANET_model_esm2_uw_orj_gercek.py`](train/PLANET_model_esm2_uw_orj_gercek.py) | `ABL2_P0_700k/PLANET.iter-700000` | ESM-2 embeddings replace the BLOSUM62 features |
| **PLIANT-ESM-Pose** | [`train/PLANET_model_esm2_vg_orj_ceptemask.py`](train/PLANET_model_esm2_vg_orj_ceptemask.py) | `BYER0_700k/PLANET.iter-700000` | adds cross-modal gating and distance-biased vector attention |

Shared layers: [`train/layers_esm2_uw_orj.py`](train/layers_esm2_uw_orj.py) for the
first two, [`train/layers_esm2_vg_orj.py`](train/layers_esm2_vg_orj.py) for the
third. Training driver: [`train/PLANET_train_abl.py`](train/PLANET_train_abl.py).
The 140k→700k continuation on the cluster was launched with
[`train/sb_kont_truba.sh`](train/sb_kont_truba.sh).

**The first two variants are pose-blind by construction.** The only path from
ligand coordinates to the predicted score is the distance bias inside vector
attention (`ProteinLigandAttention.geo_bias`). Without it the model cannot
distinguish two poses of the same ligand — this is an architectural property, and
it is what the docking-power and pose-margin results measure.

## In-pocket decoys and the two margins

Every non-binding molecule is moved into the pocket instead of being left far
outside it:

| Step | File |
|---|---|
| Placement core — 30 random orientations, clash penalty from van der Waals radii, contact reward, then local relaxation with AutoDock Vina | [`decoys/cepte_yerlestir3.py`](decoys/cepte_yerlestir3.py) |
| Driver that walks a split and writes the placed poses | [`decoys/yerlesim_gecis4.py`](decoys/yerlesim_gecis4.py) |
| Builds the LMDB consumed by training | [`decoys/lmdb_kur_cepte.py`](decoys/lmdb_kur_cepte.py) |
| Launchers (the split is passed as `SPLIT=` at submit time) | [`decoys/sb_yer4.sh`](decoys/sb_yer4.sh), [`decoys/sb_yer_vt.sh`](decoys/sb_yer_vt.sh), [`decoys/sb_lmdb_cepte2.sh`](decoys/sb_lmdb_cepte2.sh), [`decoys/sb_lmdb_vt_cepte.sh`](decoys/sb_lmdb_vt_cepte.sh) |

The two margin losses live in the `compute_loss` of each model module above: a
binder must score at least 1 pK above a decoy in the same pocket, and a
crystallographic pose at least 1 pK above the same ligand displaced by 1–3 Å.

## Which script produced which reported number

| Reported value | Produced by | Launcher |
|---|---|---|
| CASF-2016 Pearson *r* (0.824 / 0.809 / 0.844) and MAE (1.004 / 0.973 / 0.920) | the training run's own evaluation on the 271-complex core LMDB | [`train/PLANET_train_abl.py`](train/PLANET_train_abl.py) |
| CASF-2016 contact-map AUROC (0.866 / 0.880 / 0.958) | the same training-time evaluation (`Test/Contact_AUROC`) | [`train/PLANET_train_abl.py`](train/PLANET_train_abl.py) |
| — independent check of the above, binder-only, 284 targets | [`eval/eval_temas_binder.py`](eval/eval_temas_binder.py) | [`eval/sb_temas_700k.sh`](eval/sb_temas_700k.sh) |
| CASF-2016 docking power (24.4 / 24.4 / 71.5 %) | [`eval/eval_dp_v2.py`](eval/eval_dp_v2.py) | [`eval/sb_dp_v2_b700.sh`](eval/sb_dp_v2_b700.sh), [`eval/sb_dp_v2_uw700.sh`](eval/sb_dp_v2_uw700.sh), [`eval/sb_abl1_dp.sh`](eval/sb_abl1_dp.sh) |
| DUD-E, 101 targets, centred (0.544 / 0.640 / 0.545) | [`eval/DUDE_dualchannel_skor.py`](eval/DUDE_dualchannel_skor.py) | [`eval/dude_abl.sh`](eval/dude_abl.sh) with `MERKEZ=1` |
| LIT-PCBA, 15 targets, centred (0.499 / 0.476 / 0.548) | [`eval/LITPCBA_dualchannel.py`](eval/LITPCBA_dualchannel.py) | [`eval/litpcba_abl.sh`](eval/litpcba_abl.sh) with `MERKEZ=1` |

Two details worth knowing when reading these scripts:

- **`MERKEZ=1` is the centred protocol.** It passes `--center_ligands`, which
  moves each ligand's centre onto the pocket centre with no rotation and no clash
  correction. `MERKEZ=0` is the uncentred protocol used elsewhere in this project
  and is *not* what the abstract reports.
- **Docking power ranks poses by the predicted affinity.** `eval_dp_v2.py` also
  computes contact-map agreement channels; the affinity channel is the one
  reported. Ties are broken at random and averaged over 200 repeats (`N_TIE=200`,
  `RMSD_CUT=2.0`), because a pose-blind model assigns every pose the same score
  and would otherwise score far above chance by always picking the first pose in
  the file.

## What is not here

- **Trained weights.** `PLANET.iter-700000` checkpoints are not in git; see the
  checkpoint note in the repository root.
- **Data.** The LMDBs with in-pocket decoys are large and hosted separately; they
  are not the thesis LMDBs listed in the root README.
- **Analyses not reported in the abstract.** Pocket-swap specificity, docked-pose
  screening and the contrastive variants were measured but are not part of the
  abstract, so their scripts are not included here.

## Notes

- Module and checkpoint names keep the upstream `PLANET_*` prefix on purpose; see
  "Relation to PLANET" in the repository root README.
- The `.sh` files are an exact record of how the jobs were submitted on the
  TRUBA/ARF cluster (partition, account, array layout) and contain cluster paths.
  They are not portable as-is; the `.py` scripts they call are the reusable part.
- Some runs started on a local GPU workstation and were continued on the cluster
  from a saved checkpoint; `sb_kont_truba.sh` is the continuation launcher.
