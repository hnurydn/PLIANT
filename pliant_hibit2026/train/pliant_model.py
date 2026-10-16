import os, sys
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════
# [ABLATION ARM 1] Baseline + uncertainty weighting (no ESM-2, no vector attention, no gating).
#
# Architecture is IDENTICAL to pliant_esm_model (arm 2); the only difference is the residue input:
#     ESM-2 (512 dims)  ->  BLOSUM62 (20 dims)
#     res_embedding = Linear(20, d, bias=False) -> LeakyReLU(0.1)   (no LayerNorm)
# This matches the thesis baseline exactly. The LayerNorm(512) present in arm 2 belongs to the
# ESM input and arrives together with ESM in the 1 -> 2 step.
#
# DATA PIPELINE PITFALL: chemutils_esm2_vector.tensorize_protein_pocket pads a feature vector
# shorter than FEATURE_DIM (512) with zeros WITHOUT raising. Importing this module sets
# FEATURE_DIM = 20 (the training script imports the model BEFORE the DataLoader workers fork,
# so the workers see 20 as well).
# The same function also silently TRUNCATES a longer feature vector, so feeding 512-dim ESM data
# by mistake would proceed with the first 20 ESM dimensions. The guard is therefore on the DATA
# side: the launcher checks that the LMDB is the ..._blosum20 one, that DONUSUM.json is present,
# and that the first pockets really do carry 20-element features.
#
# Loss, compute_loss and the uncertainty weighting (log_sigma) are identical to arm 2 and to the
# pose-aware arm.
# ═══════════════════════════════════════════════════════════════════════════
import chemutils_esm2_vector as _CU
REZIDU_BOYUT = 20
_CU.FEATURE_DIM = REZIDU_BOYUT

from pliant_esm_model import PLANET as _PLANET_KOL2


class PLANET(_PLANET_KOL2):
    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters,
                 lig_update_iters, pro_lig_update_iters, device):
        super().__init__(feature_dims, nheads, key_dims, value_dims, pro_update_inters,
                         lig_update_iters, pro_lig_update_iters, device)
        self.proteinegnn.res_embedding = nn.Sequential(
            nn.Linear(REZIDU_BOYUT, feature_dims, bias=False),
            nn.LeakyReLU(0.1),
        ).to(self.device)
