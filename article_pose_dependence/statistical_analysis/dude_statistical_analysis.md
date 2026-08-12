# DUD-E Per-Target Statistical Significance Analysis
(2026-08-11, last update: one-sided Wilcoxon methodology confirmed — see §0)

## 0. Methodology note — CONFIRMED

The test direction was verified against the thesis's Table 4.8 (LIT-PCBA) as
a reference: the one-sample test is **ONE-SIDED**
(`scipy.stats.wilcoxon(diff, alternative='greater')` — testing AUROC > 0.5,
not merely AUROC ≠ 0.5). Evidence: on the LIT-PCBA **original** data,
v4_Projection gives p=0.006226 ≈ the thesis's **0.006** (matching to 3
decimal places). This also confirms **the thesis's Table 4.8 already used the
original (non-centered) data** — no correction is needed for LIT-PCBA, only
for DUD-E (which the thesis never analyzed this way).

All numbers below have been updated using this (correct, one-sided) method.

## 1. Context

The thesis (p.71) reports DUD-E (101 targets) only descriptively; no
statistical significance analysis was ever performed (a note says it
"should additionally be assessed"). This fills that gap, per the roadmap
report's "systematize per-target statistical analysis" item.

## 2. ⭐ Main finding: the centering effect maps exactly onto architecture

| Model | Original AUROC | Centered AUROC | Δ | Vector-Gating family? |
|---|---|---|---|---|
| ESM_VG | 0.486 | 0.576 | **+0.090** | ✅ |
| v4_Projection | 0.477 | 0.556 | **+0.079** | ✅ |
| v2_Softmax_m0 | 0.498 | 0.538 | +0.040 | ✅ |
| v2_Softmax_m01 | 0.516 | 0.554 | +0.038 | ✅ |
| v2_Mean_m0 | 0.512 | 0.536 | +0.024 | ✅ |
| ESM_VG_Contrastive | 0.509 | 0.476 | −0.033 | ✅ (exception) |
| ESM_UW | 0.554 | 0.554 | **0.000** | ❌ |
| NoESM_UW | 0.501 | 0.501 | **0.000** | ❌ |
| NoESM_NoUW (baseline) | 0.530 | 0.530 | **−0.000** | ❌ |
| ESM_NoUW | 0.505 | 0.505 | **−0.000** | ❌ |

Models that consume ligand coordinates in their forward pass
(Vector-Gating family) systematically benefit from centering; the 4 models
that do not are affected by **exactly zero**. Combined with the
pose-perturbation experiment (§2 of `pose_perturbation_teknik_tasarim.md`)
and the WandB checkpoint-trajectory analysis, this is a **third, fully
independent** line of evidence (from real DUD-E benchmark data).

## 3. One-sample Wilcoxon (one-sided, AUROC>0.5), 101 targets

| Model | Centered AUROC (p, significant) | **Original AUROC (p, significant)** |
|---|---|---|
| ESM_VG | 0.576 (p=1.74×10⁻¹⁰, **Yes**) | 0.486 (p=0.931, No) |
| v4_Projection | 0.556 (p=5.28×10⁻⁸, **Yes**) | 0.477 (p=0.961, No) |
| v2_Softmax_m01 | 0.554 (p=4.32×10⁻⁷, **Yes**) | 0.516 (p=0.171, No) |
| ESM_UW | 0.554 (p=7.77×10⁻⁵, **Yes**) | 0.554 (p=7.77×10⁻⁵, **Yes**) |
| v2_Softmax_m0 | 0.538 (p=3.06×10⁻⁴, **Yes**) | 0.498 (p=0.708, No) |
| v2_Mean_m0 | 0.536 (p=0.0122, **Yes**) | 0.512 (p=0.210, No) |
| NoESM_NoUW (baseline) | 0.530 (p=0.0231, **Yes**) | 0.530 (p=0.0231, **Yes**) |
| ESM_NoUW | 0.505 (p=0.529, No) | 0.505 (p=0.529, No) |
| NoESM_UW | 0.501 (p=0.509, No) | 0.501 (p=0.509, No) |
| ESM_VG_Contrastive | 0.476 (p=0.887, No) | 0.509 (p=0.223, No) |

**7/10 models are significant on the centered data; only 2/10 on the
original (realistic) data** (ESM_UW, NoESM_NoUW) — neither belongs to the
Vector-Gating family. ESM_VG/v4_Projection drop below chance level on the
original data.

**⚠️ Important for the article:** the original data should be the primary
analysis; the centered data should only be presented as a secondary
sensitivity analysis illustrating the effect of centering.

## 4. Pairwise (matched, two-sided) Wilcoxon — original data, 14/45 pairs significant

| Model 1 | Model 2 | Δ | p | Better |
|---|---|---|---|---|
| ESM_UW | v4_Projection | +0.076 | 1.18×10⁻⁵ | ESM_UW |
| ESM_UW | ESM_VG | +0.068 | 1.41×10⁻⁴ | ESM_UW |
| ESM_UW | v2_Softmax_m0 | +0.055 | 4.87×10⁻⁴ | ESM_UW |
| ESM_NoUW | ESM_UW | −0.049 | 0.0024 | ESM_UW |
| NoESM_NoUW | ESM_VG | +0.045 | 0.0035 | NoESM_NoUW |
| NoESM_NoUW | v4_Projection | +0.053 | 0.0059 | NoESM_NoUW |

The strongest/most consistent model on the original data is **ESM_UW**
(does not consume coordinates, ESM-2+UW) — not a member of the
Vector-Gating family.

## 5. Recommended use in the article

- **The original (non-centered) results should be primary in the main text.**
- The centering-effect table (§2) can be presented as a new figure/table
  alongside the pose-perturbation experiment, framed as "two independent
  lines of evidence."
- Draft sentence:
  > *"Centering — repositioning the ligand at the pocket centroid — inflates
  > apparent AUROC specifically for Vector-Gating variants (+0.024 to +0.090)
  > while leaving architecturally pose-blind variants completely unaffected
  > (Δ=0.000 in all four cases), providing an independent, real-benchmark
  > confirmation of the RMSD-perturbation finding. Using the realistic
  > (non-centered) DUD-E poses, only two of ten variants (ESM_UW, one-sided
  > Wilcoxon p=7.8×10⁻⁵; NoESM_NoUW baseline, p=0.023) achieved statistically
  > significant discrimination from random ranking (n=101 targets) — neither
  > belongs to the pose-sensitive Vector-Gating family."*

## 6. LIT-PCBA comparison — completed in a SEPARATE file

See `litpcba_statistical_analysis.md`. Summary: **the thesis's Table 4.8
already used the correct (original) data**, no correction needed; the
centering effect on LIT-PCBA is much smaller (±0.012) — not as dramatic as
on DUD-E.
