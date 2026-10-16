import os, sys
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════
# [ABLASYON KOL 1 — 2 Eki 2026] Baseline + UW  (ESM YOK, vector attention YOK, gating YOK)
# = pliant_esm_model (KOL 2) ile AYNI mimari; TEK fark rezidu girisi:
#     ESM2 (512)  ->  BLOSUM62 (20)   [veri: lmdb_planet_v2_cepte_prot_byer_blosum20]
#   res_embedding = Linear(20, d, bias=False) -> LeakyReLU(0.1)   (LayerNorm YOK)
#   3 Eki kullanici karari: tez baseline'i (Scripts/layers.py:263) ile BIREBIR. Kol 2'deki
#   LayerNorm(512) ESM girisine ozgu; 1->2 gecisinde ESM ile birlikte gelir (tezdeki baseline<->ESM farki gibi).
# ⛔ VERI HATTI TUZAGI: chemutils_esm2_vector.tensorize_protein_pocket, FEATURE_DIM (512)'den kisa
#   ozniteligi HATA VERMEDEN sifirla 512'ye tamamliyor. Bu modul import edildiginde FEATURE_DIM=20
#   yapilir (egitim betigi modeli DataLoader iscileri catallanmadan ONCE import eder -> isciler de 20 gorur).
#   ⚠️ Ayni fonksiyon UZUN ozniteligi de sessizce KIRPAR (yanlislikla 512'lik ESM verisi verilirse ilk 20
#   ESM boyutuyla devam eder) -> koruma VERI tarafinda: baslatici/kapi LMDB'nin ..._blosum20 oldugunu,
#   DONUSUM.json'u ve ilk ceplerin ozniteliginin gercekten 20 eleman oldugunu sinar.
# Kayip / compute_loss / UW (log_sigma): kol 2 ve BYER0 ile BIREBIR.
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
