# PLANET + Ligand Language Model Extension

An extension adding an 11th model variant to the 10 model variants compared
in the thesis/JCIM article: a **ChemBERTa** (SMILES-based language model)
embedding is added on the ligand side, fused into the existing `LigandGAT`
output via a `LigandLMGate` module using the same Cross-Modal Gating (CMG)
pattern already used on the protein side. Goal: test whether enriching the
ligand side (mirroring what ESM-2 does for the protein side) also closes the
pose-independent generalization gap identified in the article.

Everything here targets **3 representative models** (not all 10) — rationale:
baseline/ESM_VG/v4_Projection are the same 3 models already used in the
pose-perturbation experiment, spanning all 4 architectural axes, at ~30% of
the cost/time of covering all 10:

| Model | Folder | Base architecture file (elsewhere in the repo, unmodified) |
|---|---|---|
| Baseline (NoESM_NoUW) | `baseline_ligandlm/` | `Scripts/models/baseline_noESM_noUW/layers.py` |
| ESM_VG | `esm_vg_ligandlm/` | `Scripts/models/ESM_VG/layers_esm2_vector_gating.py` |
| v4_Projection | `v4_projection_ligandlm/` | `Scripts/models/v4_Projection/layers_esm2_vector_gating_projection_head.py` |

## Folder structure

```
data_preparation/
  10_lmdb_v3_chemberta.py   — the ONE real data-preparation script. Reads
                               lmdb_planet_v2 (binder+decoy), computes a
                               ChemBERTa embedding for every ligand, writes
                               the result into a new lmdb_planet_v3.

{baseline,esm_vg,v4_projection}_ligandlm/
  layers_ligand_lm_gate.py               — shared gate module (3 identical copies)
  layers*.py / nnutils.py                — copied from Scripts/models/<model>/ so
                                            each folder is self-contained
  chemutils*_ligandlm.py                 — copy of the base chemutils with a 7th
                                            element (ChemBERTa embedding) added
                                            to tensorize_ligand()
  PLANET_datautils*_ligandlm.py          — only the import line changed
  PLANET_model*_ligandlm.py              — gate injected right after LigandGAT
  PLANET_train_*_ligandlm.py             — training loop (WandB project: PLANET_LIGAND_LM)
  launch_*_ligandlm.sh                   — TRUBA sbatch script (barbun-cuda)

anzu/
  launch_anzu_ligandlm.sh   — launches 2 models on a single GPU on Anzu (no
                               Slurm there) via nohup, so as not to occupy
                               more than one GPU that other users might need
```

**Important — none of the original 10 models' files were modified.**
Everything here is a new `_ligandlm`-suffixed copy (files like
`chemutils_esm2_vector.py` are also used by other eval/benchmark scripts, so
editing them in place would have broken those).

## Abandoned intermediate approaches (kept here for the record)

Two intermediate scripts were written and then **fully abandoned** once a
much faster approach was found, so they are intentionally NOT included in
this folder:
- `09_generate_ChemBERTa_embeddings.py` (binder-only, h5 cache) — first
  attempt, precomputed only binder embeddings.
- `09_generate_ChemBERTa_embeddings_decoys.py` (decoy embeddings read from
  `all_pickles/`, ~441GB of I/O, estimated ~4.5 days) — abandoned once it
  became clear `lmdb_planet_v2` (~25GB, a single memory-mapped file, ~100x
  faster) could be used instead.

`10_lmdb_v3_chemberta.py` does the job of both scripts in a **single pass**,
much faster — it computes both binder and decoy embeddings directly from each
`lmdb_planet_v2` record and writes the result into the new `lmdb_planet_v3`
record. There is no need to keep the two abandoned scripts for
reproducibility.

## Status (as of 2026-08-12)

- `lmdb_planet_v3` generation is in progress (TRUBA, Job 6230180 + chained
  continuation job 6230376)
- None of the 3 models have been trained yet — the scripts in this folder
  are ready and will be launched once `lmdb_planet_v3` finishes (ESM_VG +
  baseline → single GPU on Anzu, v4_Projection → TRUBA)
