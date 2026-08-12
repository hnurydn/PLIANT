# PLANET + Ligand Language Model Extension

Tez/JCIM makalesindeki 10 model varyantına 11. bir varyant ekleme çalışması:
ligand tarafına **ChemBERTa** (SMILES tabanlı dil modeli) embedding'i ekleyip,
CMG (Cross-Modal Gating) ile aynı desende `LigandLMGate` modülüyle mevcut
`LigandGAT` çıktısına karıştırıyoruz. Amaç: protein tarafını zenginleştirmenin
(ESM-2) yaptığı gibi, ligand tarafını zenginleştirmek de pose-bağımsız
genelleşme açığını kapatıyor mu, sorusunu test etmek.

Bu klasördeki her şey **3 temsilci model** için hazırlandı (10 modelin
hepsi değil — gerekçe: baseline/ESM_VG/v4_Projection zaten pose-perturbation
deneyinde kullanılan aynı 3 model, 4 mimari ekseni kapsıyor, kaynak/süre
maliyeti 10 modelin ~%30'u):

| Model | Klasör | Temel mimari dosyası (repo'nun geri kalanında, değiştirilmedi) |
|---|---|---|
| Baseline (NoESM_NoUW) | `baseline_ligandlm/` | `Scripts/layers.py` |
| ESM_VG | `esm_vg_ligandlm/` | `Scripts/layers_esm2_vector_gating.py` |
| v4_Projection | `v4_projection_ligandlm/` | `Scripts/layers_esm2_vector_gating_projection_head.py` |

## Klasör yapısı

```
data_preparation/
  10_lmdb_v3_chemberta.py   — TEK gerçek veri hazırlama script'i. lmdb_planet_v2
                               (binder+decoy) okuyup her ligand icin ChemBERTa
                               embedding hesaplar, yeni lmdb_planet_v3'e yazar.

{baseline,esm_vg,v4_projection}_ligandlm/
  layers_ligand_lm_gate.py               — paylaşılan gate modülü (3 kopya, aynı dosya)
  chemutils*_ligandlm.py                 — tensorize_ligand'a 7. eleman (chemberta
                                            embedding) ekleyen kopya
  PLANET_datautils*_ligandlm.py          — sadece import satırı değişti
  PLANET_model*_ligandlm.py              — gate LigandGAT'ten sonra enjekte edildi
  PLANET_train_*_ligandlm.py             — eğitim döngüsü (WandB proje: PLANET_LIGAND_LM)
  launch_*_ligandlm.sh                   — TRUBA sbatch script'i (barbun-cuda)

anzu/
  launch_anzu_ligandlm.sh   — Anzu'da (Slurm yok) tek GPU'da 2 model (ESM_VG+
                               baseline) nohup ile başlatan script
```

**Önemli — orijinal 10 modelin dosyalarına HİÇ dokunulmadı.** Her şey `_ligandlm`
son ekli yeni kopyalar olarak eklendi (`chemutils_esm2_vector.py` gibi paylaşılan
dosyalar aynı zamanda diğer eval/benchmark script'leri tarafından kullanıldığı
için doğrudan değiştirilmesi onları kırardı).

## Artık kullanılmayan / terk edilen yaklaşımlar (kayıt için)

İki ara script yazıldı ama daha hızlı bir yol bulununca **tamamen terk edildi**,
bu yüzden bu klasöre eklenmedi:
- `09_generate_ChemBERTa_embeddings.py` (binder-only, h5 cache) — sadece binder
  embedding'lerini önceden hesaplayan ilk deneme.
- `09_generate_ChemBERTa_embeddings_decoys.py` (all_pickles'tan decoy embedding,
  ~441GB okuma, tahmini ~4.5 gün) — `all_pickles/` yerine `lmdb_planet_v2`'nin
  (~25GB, tek memory-mapped dosya, ~100x daha hızlı) kullanılabileceği fark
  edilince terk edildi.

`10_lmdb_v3_chemberta.py` bu ikisinin yaptığı işi TEK GEÇİŞTE, çok daha hızlı
yapıyor — hem binder hem decoy embedding'ini `lmdb_planet_v2` kaydından
doğrudan hesaplayıp yeni `lmdb_planet_v3` kaydına yazıyor. Reprodüktibilite
için bu iki terk edilmiş script'i saklamaya gerek yok.

## Durum (2026-08-12 itibarıyla)

- `lmdb_planet_v3` üretimi devam ediyor (TRUBA, Job 6230180 + zincir 6230376)
- 3 model henüz eğitilmedi — bu klasördeki script'ler hazır, eğitim `lmdb_planet_v3`
  tamamlanınca başlatılacak (ESM_VG+baseline → Anzu tek GPU, v4_Projection → TRUBA)
