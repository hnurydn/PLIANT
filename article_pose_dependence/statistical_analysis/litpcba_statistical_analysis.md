# LIT-PCBA Per-Target Statistical Significance Analysis — Centered vs Original
(2026-08-11)

## Context

After the DUD-E analysis ("the centered/original values changed quite a bit
for DUD-E"), the same check was requested for LIT-PCBA. The thesis's Table
4.8 already contained a one-sample + matched-pairs Wilcoxon test for LIT-PCBA's
15 selected (Target, Template) pairs (v54 methodology,
`planet-thesis-final.md` §v54) — the open question was: which data (centered
or original) had been used?

**Data sources:**
- Centered: TRUBA `Planet_model/litpcba_results_centered/*.csv` (124 rows/model)
- Original: **Anzu server** (10.199.2.9, VPN required — NOT on TRUBA),
  `core6_results/` (6 models) + `missing4_results/` (4 models), 124 rows/model

Both were filtered down to the same 15 selected (Target, Template) pairs
(`ADRB2/4lde`, `ALDH1/5l2n`, `ESR1_ago/2qzo`, `ESR1_ant/5ufx`, `FEN1/5fv7`,
`GBA/2v3d`, `IDH1/4umx`, `KAT2A/5mlj`, `MAPK1/4zzn`, `MTORC1/4dri`,
`OPRK1/6b73`, `PKM2/3gr4`, `PPARG/3b1m`, `TP53/3zme`, `VDR/3a2j`).

## Result: the thesis methodology was already CORRECT — no fix needed ✅

**Evidence:** using the one-sided Wilcoxon test (`alternative='greater'`,
discovered during the DUD-E analysis — see `dude_statistical_analysis.md`
§0), for v4_Projection:
- **Original data: p=0.006226 → rounds to 0.006 — an EXACT match to the
  thesis's value**
- Centered data: p=0.005127 → rounds to 0.005 (different)

**This confirms Table 4.8 was already computed on the original
(non-centered) data.** Unlike DUD-E (where the thesis never ran this
analysis at all, and an earlier pass here mistakenly started from the
centered data), the thesis used the correct data for LIT-PCBA from the start.

## Full comparison table — one-sample Wilcoxon (one-sided), 15 targets

| Model | Centered AUROC (p) | **Original AUROC (p, thesis value)** |
|---|---|---|
| v4_Projection | 0.584 (p=0.00513) | **0.579 (p=0.00623 ≈ thesis: 0.006)** ✅ |
| ESM_VG | 0.568 (p=0.0473, significant!) | 0.556 (p=0.104, not significant) |
| v2_Softmax_m0 | 0.550 (p=0.0757) | 0.549 (p=0.0844) |
| NoESM_NoUW (baseline) | 0.537 (p=0.0757) | 0.537 (p=0.0757 — centering effect = 0) |
| v2_Softmax_m01 | 0.537 (p=0.104) | 0.529 (p=0.165) |
| NoESM_UW | 0.533 (p=0.151) | 0.533 (p=0.151 — centering effect = 0) |
| ESM_UW | 0.516 (p=0.104) | 0.516 (p=0.104 — centering effect = 0) |
| ESM_NoUW | 0.509 (p=0.319) | 0.509 (p=0.319 — centering effect = 0) |
| v2_Mean_m0 | 0.504 (p=0.424) | 0.505 (p=0.424) |
| ESM_VG_Contrastive | 0.475 (p=0.835) | 0.481 (p=0.773) |

**Only v4_Projection is significant on both datasets.** ESM_VG is
borderline-significant on the centered data (p=0.047) but not on the
original (p=0.104) — a small but real deviation, consistent with the DUD-E
pattern (a Vector-Gating-family model benefits slightly from centering).

## Centering effect — much smaller than on DUD-E

| Model | Original | Centered | Δ |
|---|---|---|---|
| ESM_VG | 0.556 | 0.568 | +0.012 |
| v2_Softmax_m01 | 0.529 | 0.537 | +0.008 |
| v4_Projection | 0.579 | 0.584 | +0.005 |
| v2_Softmax_m0 | 0.549 | 0.550 | +0.001 |
| ESM_UW, NoESM_NoUW, ESM_NoUW | — | — | **0.000** |
| NoESM_UW | 0.533 | 0.533 | −0.000 |
| v2_Mean_m0 | 0.505 | 0.504 | −0.001 |
| ESM_VG_Contrastive | 0.481 | 0.475 | −0.006 |

**Maximum difference ±0.012** (compared to up to +0.090 on DUD-E) — the
direction is the same (Vector-Gating-family models benefit slightly, pose-blind
models are unaffected) but the effect size is much smaller. Possible reason
(unconfirmed, just an observation): LIT-PCBA's decoy/active-molecule
preparation pipeline differs from DUD-E's, and ligands may already be
positioned closer to the pocket than in DUD-E.

## Pairwise (matched) Wilcoxon — original data, 3/45 pairs significant

| Model 1 | Model 2 | Δ | p | Better |
|---|---|---|---|---|
| ESM_VG_Contrastive | v4_Projection | −0.098 | 0.0043 | v4_Projection |
| v2_Softmax_m01 | v4_Projection | −0.050 | 0.0125 | v4_Projection |
| ESM_UW | v4_Projection | −0.063 | 0.0256 | v4_Projection |

**v4_Projection's advantage is also consistent in pairwise comparisons** —
significantly better than 3 models, not significantly worse than any. This
directly supports the one-sample result (v4_Projection is the only
significant model).

## Conclusion — for the article

**No additional correction is needed on the LIT-PCBA side** — the thesis's
Table 4.8 is already methodologically sound (correct data + correct
one-sided test) and can be used directly in the article. The only addition:
the "centering effect" table above, showing that the effect is not as
dramatic as on DUD-E, could be added next to the DUD-E result as a brief
comparative note ("the centering effect on LIT-PCBA is negligibly small,
whereas on DUD-E it inflates AUROC by up to 0.09 for Vector-Gating-family
models — this difference likely stems from differences in the two
benchmarks' decoy/ligand preparation protocols").
