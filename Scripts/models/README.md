# PLANET — Model Klasörleri (10 Model, Tez/JCIM Makalesi Tablo 1)

Her klasör **kendi kendine yeterli**: o modeli eğitmek için gereken `chemutils`/
`layers`/`PLANET_model`/`PLANET_datautils`/`nnutils`/`PLANET_train_*` dosyalarının
hepsi içinde. Bazı dosyalar birden fazla klasörde aynen tekrar ediyor (örn.
`nnutils.py` hepsinde aynı) — bu, projenin kendi eski kuralı (her klasör
bağımsız çalışabilsin) korunarak yapıldı, `../evaluation/`, `../deprecated/`
gibi paylaşılan/artık kullanılmayan dosyalar ayrı tutuldu.

| Klasör | Makale Tablo 1'deki adı | Eğitim script'i | Mimari |
|---|---|---|---|
| `baseline_noESM_noUW/` | NoESM_NoUW (baseline) | `PLANET_train_noESM_noUW.py` | `PLANET_model.py` |
| `noESM_UW/` | NoESM_UW | `PLANET_train_Noesm2_UW.py` | `PLANET_model.py` (aynı, UW farkı train döngüsünde) |
| `ESM_NoUW/` | ESM_NoUW | `PLANET_train_esm2_noUW.py` | `PLANET_model_esm2.py` |
| `ESM_UW/` | ESM_UW | `PLANET_train_esm2.py` | `PLANET_model_esm2.py` (aynı, UW farkı train döngüsünde) |
| `ESM_VG/` | ESM_VG | `PLANET_train_esm2_vector_gating.py` | `PLANET_model_esm2_vector_gating.py` |
| `ESM_VG_Contrastive/` | ESM_VG_Contrastive | `PLANET_train_esm2_vector_gating_contrastive_loss.py` | `PLANET_model_esm2_vector_gating_contrastive_loss.py` |
| `v2_Mean_m0/` | v2_Mean_m0 | `PLANET_train_esm2_vector_gating_contrastive_loss_mean.py` | `..._contrastive_loss_mean.py` |
| `v2_Softmax_family/` | v2_Softmax_m0 **ve** v2_Softmax_m01 | `PLANET_train_esm2_vector_gating_contrastive_loss_v2.py` | `..._contrastive_loss_v2.py` |
| `v4_Projection/` | v4_Projection | `PLANET_train_esm2_vector_gating_contrastive_loss_v4.py` | `..._contrastive_loss_v4.py` (+ `layers_esm2_vector_gating_projection_head.py`, TEK farklı layers dosyası) |

`v2_Softmax_family/` için not: m0 ve m01 aynı script/dosyayı, sadece
`compute_loss()` içindeki hardcoded margin değeri elle 0.0→0.1 değiştirilerek
iki ayrı eğitim koşusu olarak üretilmiş (CLI argümanı değil). Dosyanın şu anki
hali margin=0.0 (→ m0). m01'in tam eğitim-zamanı kod hali kurtarılamadı, ama
mimari (forward/state_dict) ikisinde de aynı — checkpoint kullanımını etkilemez.

## Doğrulanmış iki düzeltme (2026-08-12 taşıma sırasında bulundu)

1. **`ESM_VG_Contrastive/PLANET_train_esm2_vector_gating_contrastive_loss.py`**
   — TRUBA'daki orijinal dosya `PLANET_model_esm2_vector_gating_contrastive_loss_v3`
   import ediyordu (30 Mart'taki eğitimden ~1 ay SONRA, muhtemelen v3 denemesi
   sırasında üzerine yazılmış). **Düzeltildi**: Anzu'daki eğitim-öncesi (22 Mart)
   pristine kopya + checkpoint dosyalarıyla (30 Mart) MD5 karşılaştırması,
   düz `PLANET_model_esm2_vector_gating_contrastive_loss.py`'nin (margin=1.0)
   doğru mimari olduğunu kesin olarak doğruladı.
2. Yukarıdaki v2_Softmax margin notu — kesin çözülmedi ama etkisi düşük
   (yalnızca training-time hiperparametre, checkpoint kullanımını etkilemiyor).

## Paylaşılan/model-spesifik olmayan script'ler

- `../evaluation/` — DUD-E/LIT-PCBA/CASF benchmark ve re-evaluate script'leri,
  birden fazla modeli değerlendirmek için kullanılıyor (model-spesifik değil).
- `../finetune_crossdocked/` — CrossDocked fine-tuning yolu, 10 modelin dışında
  ayrı bir deney.
- `../deprecated/` — `PLANET_model_esm2_vector_gating_contrastive_loss_v3.py`:
  contrastive_loss → mean → v2 → v3 → v4 zincirinde v4'e evrilirken atlanan
  ara adım, 10 modelin hiçbirinde kullanılmadı, sadece kayıt için tutuluyor.
- `../../Dataset_preparation/` — veri hazırlama hattı (02-13 numaralı adımlar),
  tüm modeller için ortak, dokunulmadı.
