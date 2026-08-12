# PLANET — Model Folders (10 Models, Thesis/JCIM Article Table 1)

Each folder is **self-contained**: it has every `chemutils`/`layers`/
`PLANET_model`/`PLANET_datautils`/`nnutils`/`PLANET_train_*` file needed to
train that specific model. Some files are exact duplicates across folders
(e.g. `nnutils.py` is identical in all of them) — this follows the project's
own pre-existing convention (each folder should be runnable on its own)
rather than something new; shared/no-longer-used files live separately in
`../evaluation/` and `../deprecated/`.

| Folder | Name in article Table 1 | Training script | Architecture |
|---|---|---|---|
| `baseline_noESM_noUW/` | NoESM_NoUW (baseline) | `PLANET_train_noESM_noUW.py` | `PLANET_model.py` |
| `noESM_UW/` | NoESM_UW | `PLANET_train_Noesm2_UW.py` | `PLANET_model.py` (same file; UW handled in the training loop) |
| `ESM_NoUW/` | ESM_NoUW | `PLANET_train_esm2_noUW.py` | `PLANET_model_esm2.py` |
| `ESM_UW/` | ESM_UW | `PLANET_train_esm2.py` | `PLANET_model_esm2.py` (same file; UW handled in the training loop) |
| `ESM_VG/` | ESM_VG | `PLANET_train_esm2_vector_gating.py` | `PLANET_model_esm2_vector_gating.py` |
| `ESM_VG_Contrastive/` | ESM_VG_Contrastive | `PLANET_train_esm2_vector_gating_contrastive_loss.py` | `PLANET_model_esm2_vector_gating_contrastive_loss.py` |
| `v2_Mean_m0/` | v2_Mean_m0 | `PLANET_train_esm2_vector_gating_contrastive_loss_mean.py` | `..._contrastive_loss_mean.py` |
| `v2_Softmax_family/` | v2_Softmax_m0 **and** v2_Softmax_m01 | `PLANET_train_esm2_vector_gating_contrastive_loss_v2.py` | `..._contrastive_loss_v2.py` |
| `v4_Projection/` | v4_Projection | `PLANET_train_esm2_vector_gating_contrastive_loss_v4.py` | `..._contrastive_loss_v4.py` (+ `layers_esm2_vector_gating_projection_head.py`, the only variant with a different `layers` file) |

Note on `v2_Softmax_family/`: m0 and m01 come from the same script/file — the
two runs were produced by manually editing the hardcoded contrastive margin
in `compute_loss()` between runs (0.0 → 0.1), not via a CLI flag. The file's
current content has margin=0.0 (→ m0). The exact training-time code for m01
could not be recovered, but the architecture (forward pass / state_dict) is
identical between the two, so this does not affect using either checkpoint.

## Two provenance issues found and resolved while reorganizing (2026-08-12)

1. **`ESM_VG_Contrastive/PLANET_train_esm2_vector_gating_contrastive_loss.py`**
   — on TRUBA, this file's import line pointed to
   `PLANET_model_esm2_vector_gating_contrastive_loss_v3`, but the checkpoint
   it corresponds to finished training on 2026-03-30 while that edit was made
   about a month later (2026-05-03), most likely while later experimenting
   with the v3 architecture. **Fixed**: cross-checked against a pristine,
   pre-training copy of the same file on a secondary workstation (Anzu,
   last modified 2026-03-22, untouched since) plus an MD5 comparison of the
   model file confirmed the correct import is the plain
   `PLANET_model_esm2_vector_gating_contrastive_loss.py` (margin=1.0); the
   import was corrected accordingly.
2. The v2_Softmax margin note above — not fully resolved, but low-impact
   (a training-time hyperparameter only, does not affect using the
   checkpoints).

## Non-model-specific scripts

- `../evaluation/` — DUD-E/LIT-PCBA/CASF benchmark and re-evaluation
  scripts, used to score multiple models, not tied to one specific model.
- `../finetune_crossdocked/` — the CrossDocked fine-tuning pathway, a
  separate experiment outside the 10-model comparison.
- `../deprecated/` — `PLANET_model_esm2_vector_gating_contrastive_loss_v3.py`:
  an intermediate step in the contrastive_loss → mean → v2 → v3 → v4 chain
  that was skipped over on the way to v4, not used by any of the 10 final
  models, kept only for the record.
- `../../Dataset_preparation/` — the shared data-preparation pipeline
  (numbered steps 02-13), common to all models, left untouched.
