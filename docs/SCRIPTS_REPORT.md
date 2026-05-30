# PLANET — Script Mimarisi ve İş Akışı Raporu

> Bu rapor, `PLANET-thesis` reposundaki **237 dosyanın** (132 `.py`, 91 `.sh`, 14 not/çıktı)
> otomatik analiziyle üretilmiştir. Amaç: tez yazımında her script'i tek tek hatırlamak/anlatmak
> zorunda kalmadan, "ne işe yarar, nereden çağrılır, hangi model varyantında kullanılır,
> benzerlerinden farkı nedir" sorularına tek yerden cevap vermek.
>
> Yapı: (1) üst-seviye anlatı ve veri akışı, (2) model "zoo" / varyant haritası,
> (3) otomatik envanter tabloları (her dosya), (4) varyantlar arası programatik diff'ler.
> Tablolar ve diff'ler doğrudan kaynak dosyalardan üretildi (yorum katmadan).

---

## 1. Genel Bakış

**PLANET**, protein–ligand bağlanma tahmini (binding affinity / virtual screening) için
bir derin öğrenme modelidir. Repo üç mantıksal katmandan oluşur:

| Katman | Klasör | Rolü |
|---|---|---|
| **Veri hazırlama** | `Dataset_preparation/` | Ham PDBbind → temiz protein/ligand → decoy üretimi → ESM2 embedding → LMDB |
| **Model & eğitim** | `Model/` | Mimari tanımları (`PLANET_model*`, `layers*`), eğitim (`PLANET_train*`), değerlendirme (`PLANET_eval*`, benchmark'lar) |
| **Temiz/konsolide sürümler** | `Scripts/` | Final deneylerde kullanılan derli toplu model/train/eval sürümleri (özellikle contrastive varyantlar ve `eval_unified`) |

Çalışma ortamı: TRUBA/ARF HPC, Slurm ile iş gönderimi, `pocketgen` conda ortamı
(decoy/pocket adımlarında ayrıca `banana` ortamı). Her `.sh` bir Slurm sarmalayıcıdır ve
karşılık gelen bir `.py`'yi çağırır.

---

## 2. Veri Hazırlama Hattı (`Dataset_preparation/`)

Dosyalar numaralandırılmış bir boru hattı oluşturur (sıra, dosya adı önekinden bellidir):

| Adım | Script | Ne yapar |
|---|---|---|
| 02 | `02_create_pkd_map.py` | PDBbind index'lerinden her PDB ID için **pKd/pKi** değerini JSON haritasına çıkarır |
| 03 | `03_prepare_protein.py` | Ham `.pdb`'den su/ligand/HETATM temizler, sadece protein zincirlerini tutar, **FASTA** üretir |
| 04 | `04_extract_binder_properties.py` | Her ligand için RDKit ile fizikokimyasal özellikler (MW, LogP, HBD/HBA, RotB, TPSA) |
| 05 | `05_generate_decoy_candidates.py` | **Property-matched decoy** adayları: bağlayıcıya fizikokimyasal benzer ama 2D yapıca farklı moleküller |
| 06 | `06_pocket_create_and_score.py` (+`_v2`) | "banana" anotasyonu: cep (pocket) oluşturma ve skorlama (`banana` conda ortamı) |
| 07 | `07_generate_decoy_sdfs*.py` | Decoy moleküllerini **SDF**'e çevirir (`_deneme` = aktif sürüm; `_gpu` Slurm varyantı) |
| 08 | `process_PDBBind_new.py` (`08_*` sh'ler) | Protein+ligand+decoy'u tek **PLANET veri kaydı** formatında birleştirir |
| 08.2 | `create_final_splits.py` + `PLANET_datautils.py` | train/valid/**CASF core** test bölünmelerini oluşturur |
| 09 | `09_generate_ESM2_embeddings*.py` | Protein dizilerinden **ESM2** embedding üretir (ana / `_crossdocked` / `_dude` / litpcba sh) |
| 10 | `10_update_pkl_esm2.py` | Mevcut `.pkl` kayıtlarına ESM2 embedding'lerini ekler (v2 = ESM2'li veri seti) |
| 11–12 | `11_generate_*_fastas.py`, `12_*` | FASTA üretimi ve CD-HIT ile **leakage temizliği** (train/test sızıntısı engelleme) |
| 13 | `13_generate_byte_indices.py` | LMDB için hızlı erişim **byte index**'leri |
| 17/19/20 | `17_convert_decoys_inplace.py`, `19_prepare_dude_pkl.py`, `20_prepare_lit_pcba_pkl.py` | Benchmark veri setleri (DUDE, LIT-PCBA) için `.pkl` hazırlığı |
| — | `create_lmdb.py`, `create_chunks_fast.py`, `pretensorize_lmdb.py`, `update_lmdb_decoys.py` | Nihai **LMDB** veri tabanını kurar / tensörleştirir (eğitimde okunan format) |
| — | `calculate_pca_params*.py` | ESM'siz hatta protein özelliklerini sıkıştırmak için **PCA** parametreleri |
| — | `test_set_preparetion.py/.ipynb` | Test seti hazırlık ve kontrol |

> Çıktı: `datasets/pdbbind2020/lmdb_planet` (ESM'siz) ve `lmdb_planet_v2` (ESM2'li) —
> ikisi de Hugging Face'te (bkz. `datasets/.../DATASET.md`).

**Yardımcı modüller (her yerde import edilir):** `chemutils*.py` (molekül/özellik çıkarımı,
graf oluşturma), `PLANET_datautils*.py` (veri kaydı/okuma), `nnutils.py` (NN yardımcıları),
`layers*.py` (model katmanları). Bunların ESM2 / vector / smooth varyantları, ilgili model
varyantıyla aynı eksende çoğaltılmıştır (farklar Bölüm 4'teki diff'lerde).

---

## 3. Model "Zoo" — Varyant Haritası

Modeller **4 ablasyon ekseni** üzerinde türetilmiştir. Dosya/klasör adları bu eksenlerin
kombinasyonudur:

| Eksen | Değerler | Anlamı |
|---|---|---|
| **Protein temsili** | `noESM` ↔ `ESM2` | Ham/PCA protein özellikleri mi, yoksa ESM2 dil-modeli embedding'i mi |
| **UW** | `UW` ↔ `noUW` | **Uncertainty Weighting** — çok-görevli kaybın (affinity + sınıflandırma) otomatik ağırlıklandırması |
| **Vector Gating** | `(yok)` ↔ `VECTOR_GATING/VG` | Özellik füzyonunda vektör-kapılama katmanı |
| **Contrastive** | `(yok)` ↔ `CONTRASTIVE` (+`smooth`, +`projection/v4`) | Bağlayıcı–decoy ayrımı için **contrastive** kayıp |

### 3.1 Eğitilen modeller ↔ script eşlemesi (adlandırma deseninden)

> Not: Eşleme, sunucudaki çıktı klasör adlarıyla (`ESM2_UW_1:11`, `noESM_noUW_1:11_512`,
> `ESM2_VG_CONTRASTIVE_v2_scratch`, `ESM2_VG_FINETUNE_AFF` …) eğitim script adları arasındaki
> birebir adlandırmaya dayanır. Kesin doğrulama için WandB run adlarına bakılabilir
> (`PLANET_TRAINING` → `PLANET_COMBINED_RESULT`). `1:11` ≈ bağlayıcı:decoy oranı, `512` ≈ boyut/varyant.

| Model (çıktı klasörü) | Eğitim script'i | Mimari `.py` |
|---|---|---|
| `noESM_*` (UW / noUW) | `PLANET_train.py`, `PLANET_train_Noesm2_UW.py`, `PLANET_train_noESM_noUW.py` | `PLANET_model.py` |
| `ESM2_NoUW_1:11` | `PLANET_train_esm2_noUW.py` | `PLANET_model_esm2.py` |
| `ESM2_UW_1:11` | `PLANET_train_esm2.py` | `PLANET_model_esm2.py` |
| `ESM2_UW_VECTOR_GATING_1:11` | `PLANET_train_esm2_vector_gating.py` | `PLANET_model_esm2_vector_gating.py` |
| `ESM2_VG_CONTRASTIVE_1b11d / _2 / v2_scratch / v3_* / v4_projection` | `PLANET_train_esm2_vector_gating_contrastive_loss.py` (+ `_v2/_v3/_v4` sh) | `PLANET_model_esm2_vector_gating_contrastive_loss.py` (+ Scripts/ `_mean/_v2/_v3/_v4`) |
| `ESM2_VG_*` (smooth) | `PLANET_train_esm2_vector_gating_smooth*.py` | `PLANET_model_esm2_vector_gating_smooth_contrastive_loss.py` |
| `ESM2_VG_FINETUNE_AFF / NOAFF` | `Scripts/PLANET_finetune_crossdocked.py` | crossdocked finetune |
| (ara varyant) | `PLANET_train_esm2_gating.py`, `PLANET_train_esm2_vector.py` | `PLANET_model_esm2_gating.py`, `PLANET_model_esm2_vector.py` |

`*_cpu` / `*_gpu` / `*_akya_cuda` son ekleri **aynı eğitimi farklı Slurm kaynaklarıyla** çalıştıran
sarmalayıcılardır (kod aynı, sadece partition/gres farklı).

### 3.2 Değerlendirme (benchmark) script'leri

| Benchmark | ESM'li | ESM'siz |
|---|---|---|
| **DUDE** | `Dataset_preparation/DUDE_benchmark_ESM.py`, `DUDE_benchmark_DEBUG.py`, `dude_gpu.py` | `DUDE_benchmark_No_ESM.py` |
| **LIT-PCBA** | `LITPCBA_benchmark_ESM.py` | `LIT-PCBA_benchmark_No_ESM.py` |
| **CASF / core** | `Model/PLANET_eval*.py`, `PLANET_eval_final.py` | — |
| **Konsolide** | `Scripts/eval_unified.py`, `Scripts/PLANET_re-evaluate_*.py` (all/contrastive/esm/noesm/vector_gating) | — |

`Scripts/PLANET_re-evaluate_*.py`, eğitilmiş tüm checkpoint'leri tek tip metrikle (AUC, EF, vb.)
yeniden skorlamak için kullanılır; `eval_unified.py` bunların birleşik sürümüdür.

---

## 4. Notlar / Yorum

- **`Scripts/` vs `Model/`:** `Model/` geliştirme sürecindeki tüm denemeleri (çok sayıda
  `_cpu/_gpu/_v2/_v3` sarmalayıcı) barındırır; `Scripts/` ise final deneylerde kullanılan
  **temizlenmiş** sürümlerdir. Tez metninde referans verirken `Scripts/` sürümleri daha derli topludur.
- **Tekrar eden dosyalar:** `chemutils.py`, `layers.py`, `nnutils.py`, `PLANET_datautils.py`,
  `PLANET_model.py` hem `Dataset_preparation/`, hem `Model/`, hem `Scripts/` altında bulunur;
  çoğu zaman aynı/çok benzer içeriktedir (her klasörün kendi kendine yeter olması için kopyalanmış).
- **Contrastive sürüm zinciri:** `contrastive_loss` → `_mean` (havuzlama değişikliği) →
  `_v2` → `_v3` → `_v4` (projection head) sırayla iyileştirmelerdir; tam satır farkları Bölüm 6'da.
- Aşağıdaki **Bölüm 5 (envanter)** ve **Bölüm 6 (diff)** otomatik üretildi; her dosyanın amacı,
  çağıran script'i, argümanları ve varyant farkları orada birebir listelidir.

---

# 5. (Otomatik) Envanter

# OTOMATIK BOLUM 1 — Slurm (.sh) script envanteri


### `(root)/` — kabuk/Slurm script'leri (10)

| Script | Partition | Süre | Çağırdığı .py | Anahtar kelimeler |
|---|---|---|---|---|
| `20_prepare_lit_pcba_pkl.sh` | barbun | 1-00:00:00 | 20_prepare_lit_pcba_pkl.py | pK, lit-pcba |
| `deneme.sh` | hamsi | 2-00:00:00 | inference.py |  |
| `fix_valid.sh` | barbun | 02:00:00 | create_lmdb.py | lmdb |
| `planet_train.sh` | akya-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating_contrastive_loss.py | contrastive, vector_gating, ESM2, esm2, gating, lmdb |
| `rclone.sh` | barbun | 1-00:00:00 | 07_generate_decoy_sdfs_deneme.py, 17_convert_decoys_inplace.py | decoy |
| `temizle.sh` | akya-cuda | 3-00:00:00 | — |  |
| `test.sh` | barbun | 00:05:00 | — |  |
| `train.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating.py | vector_gating, ESM2, esm2, gating, lmdb, decoy |
| `wandb_duzeltme.sh` | barbun | 3-00:00:00 | wandb_merge_all.py | wandb |
| `zip_file.sh` | barbun | 1-00:00:00 | — | ESM2, esm2, lmdb, pK |

### `Dataset_preparation/` — kabuk/Slurm script'leri (29)

| Script | Partition | Süre | Çağırdığı .py | Anahtar kelimeler |
|---|---|---|---|---|
| `01_prepare_datasets.sh` |  |  | — | decoy, EF, lit-pcba, dude, casf |
| `02_create_pkd_map.sh` | barbun | 1-00:00:00 | 02_create_pkd_map.py | pK |
| `03_prepare_protein.sh` | barbun | 1-00:00:00 | 03_prepare_protein.py | pK |
| `04_extract_binder_properties.sh` | barbun | 1-00:00:00 | 04_extract_binder_properties.py | pK |
| `05_generate_decoy_candidates.sh` | barbun | 04:00:00 | 05_generate_decoy_candidates.py | decoy |
| `06_pocket_create_and_score.sh` | debug | 2:00:00 | 06_pocket_create_and_score.py |  |
| `06_run_pocket_score.sh` | hamsi | 2-00:00:00 | 06_pocket_create_and_score.py |  |
| `07_generate_decoy_sdfs.sh` | debug | 02:00:00 | 07_generate_decoy_sdfs_deneme.py | decoy |
| `07_generate_decoy_sdfs_gpu.sh` | barbun-cuda | 10:00:00 | 07_generate_decoy_sdfs_deneme.py | decoy |
| `08_01_prepare_planet_data.sh` | debug | 04:00:00 | process_PDBBind_new.py | pK |
| `08_01_prepare_planet_data_gpu.sh` | barbun-cuda | 1-00:00:00 | process_PDBBind_new.py | pK |
| `08_02_prepare_planet_data.sh` | barbun-cuda | 03:00:00 | PLANET_datautils.py, create_final_splits.py | pK, casf |
| `08_prepare_planet_data_deneme.sh` | barbun | 1-00:00:00 | process_PDBBind_new.py | pK |
| `09_generate_ESM2_embeddings.sh` | barbun | 1-00:00:00 | 09_generate_ESM2_embeddings.py | ESM2, esm2, pK |
| `09_generate_ESM2_embeddings_crossdocked.sh` | barbun-cuda | 1-00:00:00 | 09_generate_ESM2_embeddings_crossdocked.py | ESM2, esm2, pK, crossdocked |
| `09_generate_ESM2_embeddings_dude.sh` | barbun | 1-00:00:00 | 09_generate_ESM2_embeddings_dude.py | ESM2, esm2, pK, dude |
| `09_generate_ESM2_embeddings_litpcba.sh` | hamsi | 3-00:00:00 | 09_generate_ESM2_embeddings_dude.py | ESM2, esm2, pK, litpcba, lit-pcba, dude |
| `10_update_pkl_esm2.sh` | hamsi | 3-00:00:00 | 10_update_pkl_esm2.py | ESM2, esm2, pK |
| `13_generate_byte_indices.sh` | barbun | 1-00:00:00 | 13_generate_byte_indices.py | pK |
| `19_prepare_dude_pkl.sh` | barbun | 1-00:00:00 | 19_prepare_dude_pkl.py | pK, dude |
| `20_prepare_lit_pcba_pkl.sh` | hamsi | 1-00:00:00 | 20_prepare_lit_pcba_pkl.py | pK, lit-pcba |
| `calculate_pca_params.sh` | hamsi | 12:00:00 | calculate_pca_params_dude.py, chemutils.py | decoy, dude |
| `check_pdb_ids.sh` | barbun | 1-00:00:00 | check_pdb_ids.py |  |
| `create_chunks_fast.sh` | hamsi | 1-00:00:00 | create_chunks_fast.py | lmdb |
| `create_lmdb.sh` | debug | 04:00:00 | create_lmdb.py, update_lmdb_decoys.py | lmdb, decoy |
| `prepare_lit_pcba_sdf.sh` | hamsi | 24:00:00 | prepare_lit_pcba_sdf.py |  |
| `pretensorize_lmdb.sh` | barbun | 1-00:00:00 | pretensorize_lmdb.py | lmdb |
| `test_set_preparetion.sh` | hamsi | 3-00:00:00 | test_set_preparetion.py |  |
| `zip_core.sh` | barbun | 0-12:00:00 | convert.py | lmdb |

### `Model/` — kabuk/Slurm script'leri (39)

| Script | Partition | Süre | Çağırdığı .py | Anahtar kelimeler |
|---|---|---|---|---|
| `PLANET_Dude_test.sh` | hamsi | 2-00:00:00 | DUDE.py, dude.py | dude |
| `PLANET_Eval_No_ESM.sh` | barbun-cuda | 3-00:00:00 | DUDE_benchmark_No_ESM.py | ESM2, esm2, no_esm, noesm, dude |
| `PLANET_LITPCA_Eval_No_ESM.sh` | barbun-cuda | 3-00:00:00 | LIT-PCBA_benchmark_No_ESM.py | no_esm, noesm, litpcba, lit-pcba |
| `PLANET_eval.sh` | hamsi | 1-00:00:00 | dude_gpu.py | ESM2, esm2, dude |
| `PLANET_eval_ESM2_cpu.sh` | hamsi | 3-00:00:00 | DUDE_benchmark_DEBUG.py | ESM2, esm2, dude |
| `PLANET_eval_LITPCBA_esm2.sh` | barbun-cuda | 3-00:00:00 | LITPCBA_benchmark_ESM.py | ESM2, esm2, EF, litpcba, lit-pcba |
| `PLANET_eval_dude.sh` | barbun-cuda | 3-00:00:00 | DUDE_benchmark_DEBUG.py, DUDE_benchmark_No_ESM.py | contrastive, vector_gating, ESM2, esm2, no_esm, noesm |
| `PLANET_eval_dude_2.sh` | barbun-cuda | 3-00:00:00 | DUDE_benchmark_ESM.py | contrastive, vector_gating, ESM2, esm2, gating, EF |
| `PLANET_eval_dude_2_cpu.sh` | hamsi | 3-00:00:00 | DUDE_benchmark_ESM.py | contrastive, vector_gating, ESM2, esm2, gating, dude |
| `PLANET_eval_dude_cpu.sh` | hamsi | 3-00:00:00 | DUDE_benchmark_DEBUG.py, DUDE_benchmark_No_ESM.py | contrastive, vector_gating, ESM2, esm2, no_esm, noesm |
| `PLANET_eval_esm2.sh` | barbun-cuda | 3-00:00:00 | DUDE_benchmark_DEBUG.py | ESM2, esm2, dude |
| `PLANET_eval_esm2_cpu_litpcba.sh` | hamsi | 3-00:00:00 | PLANET_eval_final.py | ESM2, esm2, litpcba, lit-pcba |
| `PLANET_eval_esm2_litpcba.sh` | barbun-cuda | 3-00:00:00 | PLANET_eval_final.py | ESM2, esm2, litpcba, lit-pcba |
| `PLANET_eval_lit-pcba.sh` | hamsi | 3-00:00:00 | LIT-PCBA_benchmark_No_ESM.py, LITPCBA_benchmark_ESM.py | contrastive, vector_gating, ESM2, esm2, no_esm, noesm |
| `PLANET_eval_lit-pcba_2_cpu.sh` | hamsi | 3-00:00:00 | LITPCBA_benchmark_ESM.py | contrastive, vector_gating, ESM2, esm2, gating, litpcba |
| `PLANET_eval_lit-pcba_cpu.sh` | barbun | 3-00:00:00 | LIT-PCBA_benchmark_No_ESM.py, LITPCBA_benchmark_ESM.py | contrastive, vector_gating, ESM2, esm2, no_esm, noesm |
| `PLANET_finetune_crossdocked.sh` | barbun-cuda | 3-00:00:00 | PLANET_finetune_crossdocked.py | vector_gating, ESM2, esm2, gating, lmdb, decoy |
| `PLANET_train.sh` | barbun-cuda | 3-00:00:00 | PLANET_train2.py, PLANET_train_noESM_noUW.py | noesm, pK |
| `PLANET_train_Noesm2_UW.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_Noesm2_UW.py | ESM2, esm2, noesm, lmdb, decoy |
| `PLANET_train_akya_cuda.sh` | akya-cuda | 3-00:00:00 | PLANET_train_esm2.py | ESM2, esm2, pK |
| `PLANET_train_cpu.sh` | hamsi | 3-00:00:00 | PLANET_train_esm2.py | ESM2, esm2, pK |
| `PLANET_train_esm2.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2.py | ESM2, esm2, lmdb, decoy |
| `PLANET_train_esm2_cpu.sh` | hamsi | 3-00:00:00 | PLANET_train_esm2.py | ESM2, esm2, pK |
| `PLANET_train_esm2_gating.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_gating.py | ESM2, esm2, gating, pK |
| `PLANET_train_esm2_noUW.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_noUW.py | ESM2, esm2, lmdb, decoy |
| `PLANET_train_esm2_vector.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector.py | ESM2, esm2, pK |
| `PLANET_train_esm2_vector_cpu.sh` | hamsi | 3-00:00:00 | PLANET_train_esm2_vector.py | ESM2, esm2, pK |
| `PLANET_train_esm2_vector_gating.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating.py | vector_gating, ESM2, esm2, gating, lmdb, decoy |
| `PLANET_train_esm2_vector_gating_contrastive_loss.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating_contrastive_loss.py | contrastive, vector_gating, ESM2, esm2, gating, lmdb |
| `PLANET_train_esm2_vector_gating_contrastive_loss_v2.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating_contrastive_loss_v2.py | contrastive, vector_gating, ESM2, esm2, gating, lmdb |
| `PLANET_train_esm2_vector_gating_contrastive_loss_v3.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating_contrastive_loss.py | contrastive, vector_gating, ESM2, esm2, gating, lmdb |
| `PLANET_train_esm2_vector_gating_contrastive_loss_v4.sh` | akya-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating_contrastive_loss_v4.py | contrastive, vector_gating, ESM2, esm2, gating, projection |
| `PLANET_train_esm2_vector_gating_smooth.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_esm2_vector_gating_smooth.py | vector_gating, ESM2, esm2, gating, smooth, pK |
| `PLANET_train_esm2_vector_gating_smooth_cpu.sh` | hamsi | 3-00:00:00 | PLANET_train_esm2_vector_gating_smooth.py | vector_gating, ESM2, esm2, gating, smooth, pK |
| `PLANET_train_eval.sh` | hamsi | 3-00:00:00 | DUDE_benchmark_TRACEBACK.py | dude |
| `PLANET_train_gpu.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_metrics.py | pK |
| `PLANET_train_noESM_noUW.sh` | barbun-cuda | 3-00:00:00 | PLANET_train_noESM_noUW.py | noesm, lmdb, decoy |
| `test_set_preparetion.sh` | hamsi | 3-00:00:00 | test_set_preparetion.py |  |
| `veri_sikistirma.sh` | hamsi | 12:00:00 | 11_generate_test_fastas.py |  |

### `Scripts/` — kabuk/Slurm script'leri (13)

| Script | Partition | Süre | Çağırdığı .py | Anahtar kelimeler |
|---|---|---|---|---|
| `DUDE_benchmark_centered..sh` | barbun-cuda | 12:00:00 | DUDE_benchmark_centered.py | contrastive, vector_gating, ESM2, esm2, noesm, gating |
| `LITPCBA_benchmark_centered.sh` | barbun-cuda | 12:00:00 | LITPCBA_benchmark_centered.py | contrastive, vector_gating, ESM2, esm2, noesm, gating |
| `create_crossdocked_lmdb.sh` | debug | 04:00:00 | create_crossdocked_lmdb.py | ESM2, esm2, lmdb, pK, crossdocked |
| `create_lmdb_pdbfixed.sh` | barbun | 06:00:00 | create_dude_lmdb.py, create_dude_lmdb_esm.py, create_litpcba_lmdb.py | ESM2, esm2, noesm, lmdb, pK, litpcba |
| `dude_test_noesm.sh` | barbun | 04:00:00 | DUDE_benchmark_NoESM_lmdb.py | noesm, wandb, lmdb, dude |
| `eval_unified.sh` | barbun-cuda | 12:00:00 | eval_unified.py | contrastive, vector_gating, ESM2, esm2, noesm, gating |
| `prepare_benchmark_proteins.sh` | barbun | 04:00:00 | prepare_benchmark_proteins.py | litpcba, lit-pcba, dude |
| `prepare_crossdocked_pkls.sh` | barbun | 04:00:00 | prepare_crossdocked_pkls.py | pK, crossdocked |
| `ssd_eval.sh` | barbun-cuda | 04:00:00 | PLANET_re-evaluate_all.py | vector_gating, ESM2, esm2, noesm, gating, lmdb |
| `ssd_eval_esm_uw.sh` | barbun-cuda | 04:00:00 | PLANET_re-evaluate_esm.py | ESM2, esm2, lmdb |
| `ssd_eval_esm_uw_vector_gating.sh` | barbun-cuda | 04:00:00 | PLANET_re-evaluate_vector_gating.py | vector_gating, ESM2, esm2, gating, lmdb |
| `ssd_eval_esm_uw_vector_gating_contrastive.sh` | barbun-cuda | 04:00:00 | PLANET_re-evaluate_contrastive.py | contrastive, vector_gating, ESM2, esm2, gating, lmdb |
| `ssd_eval_noesm_uw.sh` | barbun-cuda | 04:00:00 | PLANET_re-evaluate_noesm.py | noesm, lmdb |


# OTOMATIK BOLUM 2 — Python (.py) script envanteri


### `(root)/` — python dosyaları (5)

| Script | Satır | Amaç (docstring/yorum) | Sınıf/fonksiyon | Yerel import | Çağıran (.sh / .py) |
|---|---|---|---|---|---|
| `create_patched_chemutils.py` | 46 | — | main | — | — |
| `debug_lit_pcba.py` | 93 | — | main | chemutils_dude | — |
| `dosya_tasima.py` | 30 | — | — | — | — |
| `wandb_duzeltme.py` | 334 | — | — | — | — |
| `wandb_merge_all.py` | 195 | WandB PLANET runs merger ======================== Aynı modelin farklı checkpoint'lerden devam edilmiş runlarını NEW_PROJECT içinde tek bir "unified" run'a birle | log, fetch_run_df, process_model, main | — | wandb_duzeltme.sh |

### `Dataset_preparation/` — python dosyaları (59)

| Script | Satır | Amaç (docstring/yorum) | Sınıf/fonksiyon | Yerel import | Çağıran (.sh / .py) |
|---|---|---|---|---|---|
| `02_create_pkd_map.py` | 161 | 02_create_pkd_map.py | convert_to_pkd, load_pdbbind_index, create_structure_map | — | 02_create_pkd_map.sh |
| `03_prepare_protein.py` | 151 | 03_prepare_protein.py | prepare_protein_for_planet, process_map_for_proteins | — | 03_prepare_protein.sh |
| `04_extract_binder_properties.py` | 132 | 04_extract_binder_properties.py | get_ligand_properties, main | — | 04_extract_binder_properties.sh |
| `05_generate_decoy_candidates.py` | 356 | 05_generate_decoy_candidates.py | init_worker, get_properties_smiles_id, read_one_molecule_block, process_chunk, create_chembl_index | — | 05_generate_decoy_candidates.sh |
| `06_pocket_create_and_score.py` | 186 | — | get_ligand_centroid, parse_pdb_atom_line, create_pocket_pdb_from_data, run_smart_inference_task, main_pipeline | — | 06_pocket_create_and_score.sh, 06_run_pocket_score.sh |
| `06_pocket_create_and_score_v2.py` | 186 | — | init_banana_worker, get_ligand_centroid, parse_pdb_atom_line, create_pocket_pdb_from_data, run_inference_task | — | — |
| `07_generate_decoy_sdfs.py` | 235 | 07_generate_decoy_sdfs.py | combine_all_filtered_decoys, generate_sdf_for_row_to_pdb_dir, main | — | — |
| `07_generate_decoy_sdfs_deneme.py` | 210 | — | combine_and_get_data, process_pdb_group, main | — | 07_generate_decoy_sdfs.sh, 07_generate_decoy_sdfs_gpu.sh, rclone.sh |
| `09_generate_ESM2_embeddings.py` | 288 | 09_generate_ESM2_embeddings.py | parse_fasta_from_pdb, extract_all_sequences, compute_esm2_embeddings, main | — | 09_generate_ESM2_embeddings.sh |
| `09_generate_ESM2_embeddings_crossdocked.py` | 224 | 09_generate_ESM2_embeddings_crossdocked.py ========================================== CrossDocked 2020 receptor PDB dosyalarından ESM2 embedding üretir. Mevcut  | parse_sequence_from_pdb, build_pdb_map, generate_embeddings, main | — | 09_generate_ESM2_embeddings_crossdocked.sh |
| `09_generate_ESM2_embeddings_dude.py` | 139 | — | get_clean_id, run_esm2_generation | — | 09_generate_ESM2_embeddings_dude.sh, 09_generate_ESM2_embeddings_litpcba.sh |
| `10_update_pkl_esm2.py` | 210 | — | calculate_pca_whitening_params, load_or_calculate_pca_params, apply_pca_transform, parse_fasta_from_pdb, update_molecule_features | — | 10_update_pkl_esm2.sh |
| `11_generate_all_fastas.py` | 204 | — | get_sequence_from_pdb, get_sequence_from_mol2, process_pdbbind, process_test_sets | — | — |
| `11_generate_test_fastas.py` | 193 | — | get_sequence_from_pdb, get_sequence_from_mol2, process_test_sets | — | veri_sikistirma.sh |
| `12_generate_lit_pcba_fasta_v4.py` | 68 | — | get_multi_chain_sequence_pdb, get_multi_chain_sequence_mol2, main | — | — |
| `12_run_cdhit_clean_leakage.py` | 140 | — | parse_clstr_file, main | — | — |
| `13_generate_byte_indices.py` | 72 | — | reformat_and_index_decoys, main | — | 13_generate_byte_indices.sh |
| `17_convert_decoys_inplace.py` | 127 | — | process_record, main | chemutils | rclone.sh |
| `19_prepare_dude_pkl.py` | 132 | — | load_sdf_gz, process_target, process_wrapper, main | chemutils_dude | 19_prepare_dude_pkl.sh |
| `20_prepare_lit_pcba_pkl.py` | 137 | — | load_sdf, process_target, process_wrapper, main | chemutils_dude | 20_prepare_lit_pcba_pkl.sh |
| `DUDE.py` | 229 | — | VirtualScreening_Dataset, predict, workflow, write_csv, EF_sort | PLANET_model, chemutils | PLANET_Dude_test.sh |
| `DUDE_benchmark.py` | 261 | — | RawSDFDataset, load_sdf_gz_iterator, safe_collate_fn, run_inference, calculate_metrics | PLANET_model, chemutils_dude | — |
| `DUDE_benchmark_DEBUG.py` | 447 | — | Residue, PocketProcessor, LigandDataset, load_pca_params, apply_pca_transform_single | PLANET_model_esm2 | PLANET_eval_ESM2_cpu.sh, PLANET_eval_dude.sh, PLANET_eval_dude_cpu.sh, PLANET_eval_esm2.sh |
| `DUDE_benchmark_ESM.py` | 403 | — | Residue, PocketProcessor, LigandDataset, load_pca_params, apply_pca_transform_single | PLANET_model_esm2 | PLANET_eval_dude_2.sh, PLANET_eval_dude_2_cpu.sh |
| `DUDE_benchmark_ESM2_FULL.py` | 503 | — | Residue, Mol, ComplexPocket, ESMDataset, load_pca_params | PLANET_model_esm2 | — |
| `DUDE_benchmark_MONOLITH.py` | 460 | — | Residue, Mol, ComplexPocket, RawSDFDataset, onek_encoding_unk | PLANET_model | — |
| `DUDE_benchmark_No_ESM.py` | 286 | — | Residue, PocketProcessor, LigandDataset, move_to_device, onek_encoding_unk | PLANET_model | PLANET_Eval_No_ESM.sh, PLANET_eval_dude.sh, PLANET_eval_dude_cpu.sh |
| `DUDE_benchmark_TRACEBACK.py` | 460 | — | Residue, Mol, ComplexPocket, RawSDFDataset, onek_encoding_unk | PLANET_model | PLANET_train_eval.sh |
| `LIT-PCBA_benchmark_No_ESM.py` | 220 | — | PocketProcessor, LigandDataset, move_to_device, get_residue_feature, atom_features | PLANET_model | PLANET_LITPCA_Eval_No_ESM.sh, PLANET_eval_lit-pcba.sh, PLANET_eval_lit-pcba_cpu.sh |
| `LITPCBA_benchmark_ESM.py` | 305 | — | PocketProcessor, LigandDataset, onek_encoding_unk, atom_features, bond_features | PLANET_model_esm2 | PLANET_eval_LITPCBA_esm2.sh, PLANET_eval_lit-pcba.sh, PLANET_eval_lit-pcba_2_cpu.sh, PLANET_eval_lit-pcba_cpu.sh |
| `PLANET_datautils.py` | 180 | — | ProLigDataset, main | chemutils | 08_02_prepare_planet_data.sh, PLANET_eval.py(py), PLANET_re-evaluate_noesm.py(py), PLANET_train.py(py), PLANET_train2.py(py), PLANET_train_Noesm2_UW.py(py) |
| `PLANET_datautils_esm2.py` | 162 | — | ProLigDataset, main | chemutils_esm2 | PLANET_re-evaluate_esm.py(py), PLANET_train_esm2.py(py), PLANET_train_esm2_gating.py(py), PLANET_train_esm2_noUW.py(py), eval_unified.py(py) |
| `PLANET_datautils_esm2_vector.py` | 242 | PLANET_datautils_esm2_lmdb_v2.py ================================= lmdb_planet_v2 ile çalışır. Her binder key'i → ComplexPocket(ligand=gerçek, decoys=[Mol1, Mol | ProLigDataset | chemutils_esm2_vector | PLANET_finetune_crossdocked.py(py), PLANET_re-evaluate_all.py(py), PLANET_re-evaluate_contrastive.py(py), PLANET_re-evaluate_vector_gating.py(py), PLANET_train_esm2_vector.py(py), PLANET_train_esm2_vector_gating.py(py) |
| `PLANET_datautils_esm2_vector_new.py` | 230 | — | ProLigDataset, _merge_single_outputs | — | — |
| `PLANET_datautils_esm2_vector_smooth.py` | 160 | — | ProLigDataset, main | chemutils_esm2_vector_smooth | PLANET_train_esm2_vector_gating_smooth.py(py), PLANET_train_esm2_vector_gating_smooth_contrastive_loss.py(py) |
| `build_new_index.py` | 107 | build_new_index.py ================== train_meta.json / valid_meta.json / core_meta.json dosyalarından yeni format index pkl'leri oluşturur. Yeni index yapısı ( | build_index_from_meta, main | — | — |
| `calculate_pca_params.py` | 67 | calculate_pca_params.py | calculate_pca_whitening_params | — | — |
| `calculate_pca_params_dude.py` | 85 | — | apply_pca_transform | — | calculate_pca_params.sh |
| `check_pdb_ids.py` | 55 | — | find_missing_ids | — | check_pdb_ids.sh |
| `chemutils.py` | 457 | — | ProteinPocket, ComplexPocket, Residue, Mol, onek_encoding_unk | — | 17_convert_decoys_inplace.py(py), DUDE.py(py), PLANET_datautils.py(py), PLANET_eval.py(py), PLANET_train2.py(py), PLANET_train_Noesm2_UW.py(py) |
| `chemutils_dude.py` | 230 | — | Residue, ComplexPocket, Mol, onek_encoding_unk, atom_features | — | 19_prepare_dude_pkl.py(py), 20_prepare_lit_pcba_pkl.py(py), DUDE_benchmark.py(py), debug_lit_pcba.py(py), fix_pkl_permanently.py(py) |
| `chemutils_esm2.py` | 450 | — | ProteinPocket, ComplexPocket, Residue, Mol, onek_encoding_unk | — | PLANET_datautils_esm2.py(py), prepare_crossdocked_pkls.py(py) |
| `chemutils_esm2_vector.py` | 464 | — | ProteinPocket, ComplexPocket, Residue, Mol, onek_encoding_unk | — | PLANET_datautils_crossdocked.py(py), PLANET_datautils_esm2_vector.py(py), create_chunks_fast.py(py), pretensorize_lmdb.py(py) |
| `chemutils_esm2_vector_smooth.py` | 500 | — | ProteinPocket, ComplexPocket, Residue, Mol, gaussian_decay | — | PLANET_datautils_esm2_vector_smooth.py(py) |
| `convert.py` | 121 | build_lmdb_from_pickles.py | build_set, main | — | zip_core.sh |
| `create_chunks_fast.py` | 284 | import os, pickle, torch, sys, json import pandas as pd from tqdm import tqdm from multiprocessing import Pool # --- 1. MODÜL YÖNLENDİRMESİ --- sys.path.append( | process_pdb, main, pack_set, main, process_pdb | chemutils_esm2_vector | create_chunks_fast.sh |
| `create_final_splits.py` | 92 | — | main | — | 08_02_prepare_planet_data.sh |
| `create_lmdb.py` | 216 | update_lmdb_with_esm2.py ======================== lmdb_planet içindeki binder objelerine ESM2 embedding ekler. Her PDB için sadece 1 embedding — decoy'lar aynı  | load_pca_params, apply_pca, parse_mapping_from_pdb, inject_esm2_into_pocket, update_set | — | create_lmdb.sh, fix_valid.sh |
| `dude_gpu.py` | 291 | — | RobustPocketProcessor, VirtualScreening_Dataset, collate_fn, calculate_metrics, predict | PLANET_model, chemutils | PLANET_eval.sh |
| `fix_pkl_permanently.py` | 118 | — | fix_pocket_object, main | chemutils_dude | — |
| `inference_new_data.py` | 242 | inference_new_data.py | InferenceDataset, run_inference | PLANET_model, chemutils | — |
| `noesm_pkl_to_lmdb.py` | 117 | noesm_pkl_to_lmdb.py ==================== noESM modelleri için pkl dosyalarını LMDB'ye yazar. ComplexPocket objeleri olduğu gibi kaydedilir (tensorize edilmez). | extract_pdb_id | — | — |
| `prepare_lit_pcba_sdf.py` | 140 | — | get_centroid, get_target_center, process_smiles_to_sdf, main | — | prepare_lit_pcba_sdf.sh |
| `prepare_lmdb_data.py` | 44 | — | create_lmdb | — | — |
| `pretensorize_lmdb.py` | 122 | pretensorize_lmdb.py ==================== Mevcut LMDB'deki her ComplexPocket objesini tensorize edip yeni bir LMDB'ye yazar. Kullanım: python pretensorize_lmdb. | process_one, pretensorize | chemutils_esm2_vector | pretensorize_lmdb.sh |
| `process_PDBBind.py` | 168 | — | get_all_pdb_dirs, process_PDBBind_Complex, collect_all_pkl_paths, main | chemutils | — |
| `process_PDBBind_new.py` | 165 | — | get_all_pdb_dirs, process_PDBBind_Complex, collect_all_pkl_paths, main | chemutils | 08_01_prepare_planet_data.sh, 08_01_prepare_planet_data_gpu.sh, 08_prepare_planet_data_deneme.sh |
| `test_set_preparetion.py` | 594 | — | Residue, Mol, ComplexPocket, Args, onek_encoding_unk | — | test_set_preparetion.sh |
| `update_lmdb_decoys.py` | 116 | update_lmdb_decoys.py Belirtilen PDB listesi için LMDB'deki binder objelerinin decoy listesini günceller. | update_set, main | — | create_lmdb.sh |

### `Model/` — python dosyaları (27)

| Script | Satır | Amaç (docstring/yorum) | Sınıf/fonksiyon | Yerel import | Çağıran (.sh / .py) |
|---|---|---|---|---|---|
| `PLANET_eval.py` | 118 | — | move_to_device, safe_collate, evaluate_set | PLANET_datautils, PLANET_model, chemutils | — |
| `PLANET_eval_final.py` | 303 | — | Residue, PocketProcessor, LigandDataset, get_coords_from_line, mass_center_generic | PLANET_model_esm2 | PLANET_eval_esm2_cpu_litpcba.sh, PLANET_eval_esm2_litpcba.sh |
| `PLANET_model.py` | 128 | — | PLANET | layers, nnutils | DUDE.py(py), DUDE_benchmark.py(py), DUDE_benchmark_MONOLITH.py(py), DUDE_benchmark_No_ESM.py(py), DUDE_benchmark_TRACEBACK.py(py), DUDE_benchmark_centered.py(py) |
| `PLANET_model_esm2.py` | 128 | — | PLANET | layers_esm2, nnutils | DUDE_benchmark_DEBUG.py(py), DUDE_benchmark_ESM.py(py), DUDE_benchmark_ESM2_FULL.py(py), DUDE_benchmark_centered.py(py), LITPCBA_benchmark_ESM.py(py), LITPCBA_benchmark_centered.py(py) |
| `PLANET_model_esm2_gating.py` | 128 | — | PLANET | layers_esm2_gating, nnutils | PLANET_train_esm2_gating.py(py) |
| `PLANET_model_esm2_vector.py` | 201 | — | PLANET | layers_esm2_vector, nnutils | PLANET_train_esm2_vector.py(py) |
| `PLANET_model_esm2_vector_gating.py` | 201 | — | PLANET | layers_esm2_vector_gating, nnutils | DUDE_benchmark_centered.py(py), LITPCBA_benchmark_centered.py(py), PLANET_finetune_crossdocked.py(py), PLANET_re-evaluate_vector_gating.py(py), PLANET_train_esm2_vector_gating.py(py), PLANET_train_esm2_vector_gating_smooth.py(py) |
| `PLANET_model_esm2_vector_gating_contrastive_loss.py` | 156 | — | PLANET | layers_esm2_vector_gating, nnutils | DUDE_benchmark_centered.py(py), LITPCBA_benchmark_centered.py(py), PLANET_re-evaluate_all.py(py), PLANET_re-evaluate_contrastive.py(py), PLANET_train_esm2_vector_gating_contrastive_loss.py(py), PLANET_train_esm2_vector_gating_smooth_contrastive_loss.py(py) |
| `PLANET_model_esm2_vector_gating_smooth_contrastive_loss.py` | 222 | — | PLANET | layers_esm2_vector_gating, nnutils | — |
| `PLANET_train.py` | 293 | — | move_to_device | PLANET_datautils, PLANET_model | — |
| `PLANET_train2.py` | 202 | — | move_to_device | PLANET_datautils, PLANET_model, chemutils | PLANET_train.sh |
| `PLANET_train_Noesm2_UW.py` | 361 | PLANET_train.py (DÜZENLENMİŞ) | move_to_device | PLANET_datautils, PLANET_model, chemutils | PLANET_train_Noesm2_UW.sh |
| `PLANET_train_esm2.py` | 330 | — | move_to_device | PLANET_datautils_esm2, PLANET_model_esm2 | PLANET_train_akya_cuda.sh, PLANET_train_cpu.sh, PLANET_train_esm2.sh, PLANET_train_esm2_cpu.sh |
| `PLANET_train_esm2_gating.py` | 296 | — | move_to_device | PLANET_datautils_esm2, PLANET_model_esm2_gating | PLANET_train_esm2_gating.sh |
| `PLANET_train_esm2_noUW.py` | 322 | — | move_to_device | PLANET_datautils_esm2, PLANET_model_esm2 | PLANET_train_esm2_noUW.sh |
| `PLANET_train_esm2_vector.py` | 296 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector | PLANET_train_esm2_vector.sh, PLANET_train_esm2_vector_cpu.sh |
| `PLANET_train_esm2_vector_gating.py` | 315 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating | PLANET_train_esm2_vector_gating.sh, train.sh |
| `PLANET_train_esm2_vector_gating_contrastive_loss.py` | 306 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss | PLANET_train_esm2_vector_gating_contrastive_loss.sh, PLANET_train_esm2_vector_gating_contrastive_loss_v3.sh, planet_train.sh |
| `PLANET_train_esm2_vector_gating_smooth.py` | 296 | — | move_to_device | PLANET_datautils_esm2_vector_smooth, PLANET_model_esm2_vector_gating | PLANET_train_esm2_vector_gating_smooth.sh, PLANET_train_esm2_vector_gating_smooth_cpu.sh |
| `PLANET_train_esm2_vector_gating_smooth_contrastive_loss.py` | 296 | — | move_to_device | PLANET_datautils_esm2_vector_smooth, PLANET_model_esm2_vector_gating_contrastive_loss | — |
| `PLANET_train_noESM_noUW.py` | 312 | — | move_to_device | PLANET_datautils, PLANET_model, chemutils | PLANET_train.sh, PLANET_train_noESM_noUW.sh |
| `layers.py` | 393 | — | ProteinGAT, LigandGAT, ProLig, ProteinEGNN, ProteinLigandAttention | nnutils | PLANET_model.py(py) |
| `layers_esm2.py` | 393 | — | ProteinGAT, LigandGAT, ProLig, ProteinEGNN, ProteinLigandAttention | nnutils | PLANET_model_esm2.py(py) |
| `layers_esm2_gating.py` | 366 | — | ProteinGAT, LigandGAT, ProteinEGNN, ProteinLigandAttention, ProLig | nnutils | PLANET_model_esm2_gating.py(py) |
| `layers_esm2_vector.py` | 374 | — | ProteinGAT, LigandGAT, GaussianSmearing, ProteinLigandAttention, ProLig | nnutils | PLANET_model_esm2_vector.py(py) |
| `layers_esm2_vector_gating.py` | 426 | — | ProteinGAT, LigandGAT, GaussianSmearing, ProteinLigandAttention, ProLig | nnutils | PLANET_model_esm2_vector_gating.py(py), PLANET_model_esm2_vector_gating_contrastive_loss.py(py), PLANET_model_esm2_vector_gating_contrastive_loss_mean.py(py), PLANET_model_esm2_vector_gating_contrastive_loss_v2.py(py), PLANET_model_esm2_vector_gating_contrastive_loss_v3.py(py), PLANET_model_esm2_vector_gating_smooth_contrastive_loss.py(py) |
| `nnutils.py` | 98 | — | create_var, create_var_gpu, index_select_ND, avg_pool, stack_pad_tensor | — | PLANET_model.py(py), PLANET_model_esm2.py(py), PLANET_model_esm2_gating.py(py), PLANET_model_esm2_vector.py(py), PLANET_model_esm2_vector_gating.py(py), PLANET_model_esm2_vector_gating_contrastive_loss.py(py) |

### `Scripts/` — python dosyaları (41)

| Script | Satır | Amaç (docstring/yorum) | Sınıf/fonksiyon | Yerel import | Çağıran (.sh / .py) |
|---|---|---|---|---|---|
| `DUDE_benchmark_centered.py` | 359 | DUDE_benchmark_unified.py ========================== DUDE benchmark — tüm model tipleri desteklenir. --model_type: noesm \| esm \| vg \| contrastive | DudePocketRecord, SimpleResidue, LigandDataset, DudeLmdbReader, _get_arg | PLANET_model, PLANET_model_esm2, PLANET_model_esm2_vector_gating, PLANET_model_esm2_vector_gating_contrastive_loss | DUDE_benchmark_centered..sh |
| `LITPCBA_benchmark_centered.py` | 433 | LITPCBA_benchmark_lmdb.py ========================== LIT-PCBA benchmark — NoESM ve ESM versiyonları, LMDB tabanlı. --model_type noesm → PLANET_model (feature_di | LitPcbaPocketRecord, LigandDataset, LitPcbaLmdbReader, import_model, onek_encoding_unk | PLANET_model, PLANET_model_esm2, PLANET_model_esm2_vector_gating, PLANET_model_esm2_vector_gating_contrastive_loss | LITPCBA_benchmark_centered.sh |
| `PLANET_datautils.py` | 257 | — | ProLigDataset | chemutils | 08_02_prepare_planet_data.sh, PLANET_eval.py(py), PLANET_re-evaluate_noesm.py(py), PLANET_train.py(py), PLANET_train2.py(py), PLANET_train_Noesm2_UW.py(py) |
| `PLANET_datautils_crossdocked.py` | 229 | PLANET_datautils_crossdocked.py ================================ CrossDocked 2020 fine-tune için DataLoader. PDBbind DataLoader'dan farkları: - Her batch = aynı | CrossDockedDataset | chemutils_esm2_vector | PLANET_finetune_crossdocked.py(py) |
| `PLANET_datautils_esm2.py` | 275 | PLANET_datautils_esm2.py ========================= lmdb_planet_v2 ile çalışır. Her binder key'i → ComplexPocket(ligand=gerçek, decoys=[Mol1, Mol2, ...]) Train m | ProLigDataset | chemutils_esm2 | PLANET_re-evaluate_esm.py(py), PLANET_train_esm2.py(py), PLANET_train_esm2_gating.py(py), PLANET_train_esm2_noUW.py(py), eval_unified.py(py) |
| `PLANET_datautils_esm2_vector.py` | 337 | — | ProLigDataset | chemutils_esm2_vector | PLANET_finetune_crossdocked.py(py), PLANET_re-evaluate_all.py(py), PLANET_re-evaluate_contrastive.py(py), PLANET_re-evaluate_vector_gating.py(py), PLANET_train_esm2_vector.py(py), PLANET_train_esm2_vector_gating.py(py) |
| `PLANET_finetune_crossdocked.py` | 355 | PLANET_finetune_crossdocked.py ================================ CrossDocked 2020 verisiyle ESM_VG modelini fine-tune eder. PDBbind train script'inden farkları:  | move_to_device | PLANET_datautils_crossdocked, PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating | PLANET_finetune_crossdocked.sh |
| `PLANET_model.py` | 121 | — | PLANET | layers, nnutils | DUDE.py(py), DUDE_benchmark.py(py), DUDE_benchmark_MONOLITH.py(py), DUDE_benchmark_No_ESM.py(py), DUDE_benchmark_TRACEBACK.py(py), DUDE_benchmark_centered.py(py) |
| `PLANET_model_esm2.py` | 126 | — | PLANET | layers_esm2, nnutils | DUDE_benchmark_DEBUG.py(py), DUDE_benchmark_ESM.py(py), DUDE_benchmark_ESM2_FULL.py(py), DUDE_benchmark_centered.py(py), LITPCBA_benchmark_ESM.py(py), LITPCBA_benchmark_centered.py(py) |
| `PLANET_model_esm2_vector_gating.py` | 207 | — | PLANET | layers_esm2_vector_gating, nnutils | DUDE_benchmark_centered.py(py), LITPCBA_benchmark_centered.py(py), PLANET_finetune_crossdocked.py(py), PLANET_re-evaluate_vector_gating.py(py), PLANET_train_esm2_vector_gating.py(py), PLANET_train_esm2_vector_gating_smooth.py(py) |
| `PLANET_model_esm2_vector_gating_contrastive_loss.py` | 208 | — | PLANET | layers_esm2_vector_gating, nnutils | DUDE_benchmark_centered.py(py), LITPCBA_benchmark_centered.py(py), PLANET_re-evaluate_all.py(py), PLANET_re-evaluate_contrastive.py(py), PLANET_train_esm2_vector_gating_contrastive_loss.py(py), PLANET_train_esm2_vector_gating_smooth_contrastive_loss.py(py) |
| `PLANET_model_esm2_vector_gating_contrastive_loss_mean.py` | 186 | — | PLANET | layers_esm2_vector_gating, nnutils | PLANET_train_esm2_vector_gating_contrastive_loss_mean.py(py) |
| `PLANET_model_esm2_vector_gating_contrastive_loss_v2.py` | 219 | — | PLANET | layers_esm2_vector_gating, nnutils | PLANET_train_esm2_vector_gating_contrastive_loss_v2.py(py) |
| `PLANET_model_esm2_vector_gating_contrastive_loss_v3.py` | 221 | — | PLANET | layers_esm2_vector_gating, nnutils | PLANET_train_esm2_vector_gating_contrastive_loss.py(py) |
| `PLANET_model_esm2_vector_gating_contrastive_loss_v4.py` | 185 | — | PLANET | layers_esm2_vector_gating_projection_head, nnutils | PLANET_train_esm2_vector_gating_contrastive_loss_v4.py(py) |
| `PLANET_re-evaluate_all.py` | 143 | — | move_to_device, get_iteration, run_evaluation | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss | ssd_eval.sh |
| `PLANET_re-evaluate_contrastive.py` | 143 | — | move_to_device, get_iteration, run_evaluation | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss | ssd_eval_esm_uw_vector_gating_contrastive.sh |
| `PLANET_re-evaluate_esm.py` | 143 | — | move_to_device, get_iteration, run_evaluation | PLANET_datautils_esm2, PLANET_model_esm2 | ssd_eval_esm_uw.sh |
| `PLANET_re-evaluate_noesm.py` | 143 | — | move_to_device, get_iteration, run_evaluation | PLANET_datautils, PLANET_model | ssd_eval_noesm_uw.sh |
| `PLANET_re-evaluate_vector_gating.py` | 143 | — | move_to_device, get_iteration, run_evaluation | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating | ssd_eval_esm_uw_vector_gating.sh |
| `PLANET_train_Noesm2_UW.py` | 355 | PLANET_train.py (DÜZENLENMİŞ) | move_to_device | PLANET_datautils, PLANET_model | PLANET_train_Noesm2_UW.sh |
| `PLANET_train_esm2.py` | 316 | — | move_to_device | PLANET_datautils_esm2, PLANET_model_esm2 | PLANET_train_akya_cuda.sh, PLANET_train_cpu.sh, PLANET_train_esm2.sh, PLANET_train_esm2_cpu.sh |
| `PLANET_train_esm2_noUW.py` | 330 | — | move_to_device | PLANET_datautils_esm2, PLANET_model_esm2 | PLANET_train_esm2_noUW.sh |
| `PLANET_train_esm2_vector_gating.py` | 336 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating | PLANET_train_esm2_vector_gating.sh, train.sh |
| `PLANET_train_esm2_vector_gating_contrastive_loss.py` | 331 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss_v3 | PLANET_train_esm2_vector_gating_contrastive_loss.sh, PLANET_train_esm2_vector_gating_contrastive_loss_v3.sh, planet_train.sh |
| `PLANET_train_esm2_vector_gating_contrastive_loss_mean.py` | 328 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss_mean | — |
| `PLANET_train_esm2_vector_gating_contrastive_loss_v2.py` | 328 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss_v2 | PLANET_train_esm2_vector_gating_contrastive_loss_v2.sh |
| `PLANET_train_esm2_vector_gating_contrastive_loss_v4.py` | 330 | — | move_to_device | PLANET_datautils_esm2_vector, PLANET_model_esm2_vector_gating_contrastive_loss_v4 | PLANET_train_esm2_vector_gating_contrastive_loss_v4.sh |
| `PLANET_train_noESM_noUW.py` | 327 | — | move_to_device | PLANET_datautils, PLANET_model | PLANET_train.sh, PLANET_train_noESM_noUW.sh |
| `chemutils.py` | 447 | — | ProteinPocket, ComplexPocket, Residue, Mol, onek_encoding_unk | — | 17_convert_decoys_inplace.py(py), DUDE.py(py), PLANET_datautils.py(py), PLANET_eval.py(py), PLANET_train2.py(py), PLANET_train_Noesm2_UW.py(py) |
| `chemutils_esm2.py` | 450 | — | ProteinPocket, ComplexPocket, Residue, Mol, onek_encoding_unk | — | PLANET_datautils_esm2.py(py), prepare_crossdocked_pkls.py(py) |
| `chemutils_esm2_vector.py` | 470 | — | ProteinPocket, ComplexPocket, Residue, Mol, onek_encoding_unk | — | PLANET_datautils_crossdocked.py(py), PLANET_datautils_esm2_vector.py(py), create_chunks_fast.py(py), pretensorize_lmdb.py(py) |
| `create_crossdocked_lmdb.py` | 282 | create_crossdocked_lmdb.py =========================== CrossDocked PKL dosyalarından ESM2 embedding ekleyip LMDB oluşturur. Her binder ayrı LMDB kaydı olarak ya | load_pca_params, apply_pca, parse_mapping_from_pdb, inject_esm2, find_receptor_pdb | — | create_crossdocked_lmdb.sh |
| `eval_unified.py` | 245 | — | _get_model_type_early, move_to_device, get_iteration, collect_predictions, find_optimal_threshold | PLANET_datautils, PLANET_datautils_esm2, PLANET_datautils_esm2_vector, PLANET_model, PLANET_model_esm2, PLANET_model_esm2_vector_gating, PLANET_model_esm2_vector_gating_contrastive_loss | eval_unified.sh |
| `layers.py` | 395 | — | ProteinGAT, LigandGAT, ProLig, ProteinEGNN, ProteinLigandAttention | nnutils | PLANET_model.py(py) |
| `layers_esm2.py` | 398 | — | ProteinGAT, LigandGAT, ProLig, ProteinEGNN, ProteinLigandAttention | nnutils | PLANET_model_esm2.py(py) |
| `layers_esm2_vector_gating.py` | 402 | — | ProteinGAT, LigandGAT, GaussianSmearing, ProteinLigandAttention, ProLig | nnutils | PLANET_model_esm2_vector_gating.py(py), PLANET_model_esm2_vector_gating_contrastive_loss.py(py), PLANET_model_esm2_vector_gating_contrastive_loss_mean.py(py), PLANET_model_esm2_vector_gating_contrastive_loss_v2.py(py), PLANET_model_esm2_vector_gating_contrastive_loss_v3.py(py), PLANET_model_esm2_vector_gating_smooth_contrastive_loss.py(py) |
| `layers_esm2_vector_gating_projection_head.py` | 420 | — | ProteinGAT, LigandGAT, GaussianSmearing, ProteinLigandAttention, ProLig | nnutils | PLANET_model_esm2_vector_gating_contrastive_loss_v4.py(py) |
| `nnutils.py` | 96 | — | create_var, create_var_gpu, index_select_ND, avg_pool, stack_pad_tensor | — | PLANET_model.py(py), PLANET_model_esm2.py(py), PLANET_model_esm2_gating.py(py), PLANET_model_esm2_vector.py(py), PLANET_model_esm2_vector_gating.py(py), PLANET_model_esm2_vector_gating_contrastive_loss.py(py) |
| `prepare_benchmark_proteins.py` | 182 | prepare_benchmark_proteins.py ============================== DUDE ve LIT-PCBA protein PDB dosyalarını PDBFixer ile hazırlar. Hazırlanan dosyalar orijinal ile ay | prepare_single, collect_dude_paths, collect_litpcba_paths, update_dude_json, update_litpcba_json | — | prepare_benchmark_proteins.sh |
| `prepare_crossdocked_pkls.py` | 197 | prepare_crossdocked_pkls.py ============================ CrossDocked 2020 verisi için ComplexPocket PKL dosyaları oluşturur. PDBbind'den farkı: - Her pocket içi | find_sdf, find_receptor_pdb, process_record, main | chemutils_esm2 | prepare_crossdocked_pkls.sh |


# OTOMATIK BOLUM 3 — Eğitim/değerlendirme script argümanları


**`Dataset_preparation/09_generate_ESM2_embeddings.py`** (288 satır):
- `-m default=DEFAULT_INPUT_JSON :: Girdi JSON harita dosyası.`

**`Dataset_preparation/09_generate_ESM2_embeddings_crossdocked.py`** (224 satır):
- `--json_path :: crossdocked_new_records.json yolu`
- `--extracted_dir :: crossdocked_extracted klasörü`
- `--h5_path`
- `--batch_size default=2`

**`Dataset_preparation/09_generate_ESM2_embeddings_dude.py`** (139 satır):
- `-f default=DEFAULT_FASTA :: Girdi FASTA dosyası`
- `-o default=OUTPUT_H5_FILE :: Çıktı H5 dosyası`

**`Dataset_preparation/DUDE.py`** (229 satır):
- `-f`
- `-d`
- `-o`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Dataset_preparation/DUDE_benchmark.py`** (261 satır):
- `-m`
- `-o`
- `--batch_size default=32`
- `--device default='cuda'`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Dataset_preparation/DUDE_benchmark_DEBUG.py`** (447 satır):
- `-m`
- `-o`
- `-h5`
- `--batch_size default=64`
- `--device default='cuda'`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--out_name default='benchmark_results' :: Kaydedilecek CSV dosyasının adı`

**`Dataset_preparation/DUDE_benchmark_ESM.py`** (403 satır):
- `-m`
- `-o`
- `-h5`
- `--batch_size default=64`
- `--device default=''`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--out_name default='benchmark_results'`
- `--json_path default=DEFAULT_JSON_PATH`
- `--gating`
- `--vector`

**`Dataset_preparation/DUDE_benchmark_ESM2_FULL.py`** (503 satır):
- `-m`
- `-o`
- `-h5`
- `--batch_size default=32`
- `--device default='cuda'`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Dataset_preparation/DUDE_benchmark_MONOLITH.py`** (460 satır):
- `-m`
- `-o`
- `--batch_size default=32`
- `--device default='cuda'`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Dataset_preparation/DUDE_benchmark_No_ESM.py`** (286 satır):
- `-m`
- `-o`
- `--out_name default='benchmark_NoESM'`
- `--batch_size default=64`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Dataset_preparation/DUDE_benchmark_TRACEBACK.py`** (460 satır):
- `-m`
- `-o`
- `--batch_size default=32`
- `--device default='cuda'`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Dataset_preparation/LIT-PCBA_benchmark_No_ESM.py`** (220 satır):
- `-m`
- `-o`
- `--out_name default='benchmark_LIT_PCBA'`
- `--json_path default=DEFAULT_JSON_PATH`
- `--batch_size default=64`
- `-s default=300`

**`Dataset_preparation/LITPCBA_benchmark_ESM.py`** (305 satır):
- `-m`
- `-o`
- `-h5`
- `--json_path default=DEFAULT_JSON_PATH`
- `--batch_size default=128`
- `-s default=512`
- `--out_name default='benchmark_LIT_PCBA_ESM'`
- `--device default=''`
- `--gating`
- `--vector`

**`Dataset_preparation/PLANET_datautils.py`** (180 satır):
- `-d :: PDBbind index dizini`

**`Dataset_preparation/PLANET_datautils_esm2.py`** (162 satır):
- `-i default="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index"`

**`Dataset_preparation/PLANET_datautils_esm2_vector_smooth.py`** (160 satır):
- `-i default="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index"`

**`Dataset_preparation/create_lmdb.py`** (216 satır):
- `--src_lmdb`
- `--out_lmdb`
- `--sets default=['train'`

**`Dataset_preparation/dude_gpu.py`** (291 satır):
- `-f`
- `-o`
- `--device default='cuda'`
- `--batch_size default=128`
- `-d default=''`
- `--feature_dims default=300`
- `--nheads default=8`
- `--key_dims default=300`
- `--value_dims default=300`

**`Dataset_preparation/process_PDBBind.py`** (168 satır):
- `-d :: PDB ana dizini.`
- `-n :: CPU sayısı.`
- `-o default="all_records_for_training.pkl"`

**`Dataset_preparation/process_PDBBind_new.py`** (165 satır):
- `-d :: PDB ana dizini.`
- `-n :: CPU sayısı.`
- `-o default="all_records_for_training.pkl"`

**`Model/PLANET_eval.py`** (118 satır):
- `-v`
- `-te`
- `-t :: Gerekli değil ama hata vermemesi için eklendi`
- `-d`
- `--load_checkpoint`
- `--batch_size default=16`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`

**`Model/PLANET_eval_final.py`** (303 satır):
- `-m`
- `-o`
- `-h5`
- `-j`
- `--batch_size default=128`
- `--device default='cuda'`

**`Model/PLANET_train.py`** (293 satır):
- `-t :: Train.pkl yolu`
- `-v :: Valid.pkl yolu`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=10.0`
- `--epoch default=250`
- `--batch_size default=16`
- `--anneal_iter default=20000`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Model/PLANET_train2.py`** (202 satır):
- `-t`
- `-v`
- `-te`
- `-d`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-5`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=24`
- `--anneal_iter default=20000`
- `--initial_step default=0`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--fine_tune_affinity`

**`Model/PLANET_train_Noesm2_UW.py`** (361 satır):
- `-t :: Yol Train.pkl dosyasına.`
- `-v :: Yol Valid.pkl dosyasına.`
- `-te :: Yol Core.pkl dosyasına.`
- `-d :: Model ağırlıklarının kaydedileceği dizin.`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=16`
- `--anneal_iter default=50000`
- `--initial_step default=0`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--binder_per_batch default=2`
- `--load_checkpoint default=None`
- `--fine_tune_affinity :: Etkinleştirilirse, geri yayılımda sadece Affinity Loss kulla`

**`Model/PLANET_train_esm2.py`** (330 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512 :: ESM-2 boyutu`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=32`
- `--anneal_iter default=50000`
- `--save_iter default=5-10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`
- `--binder_per_batch default=2`

**`Model/PLANET_train_esm2_gating.py`** (296 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512 :: ESM-2 boyutu`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=32`
- `--anneal_iter default=20000`
- `--save_iter default=5-10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Model/PLANET_train_esm2_noUW.py`** (322 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512 :: ESM-2 boyutu`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=24`
- `--anneal_iter default=50000`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`
- `--binder_per_batch default=2`

**`Model/PLANET_train_esm2_vector.py`** (296 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512 :: ESM-2 boyutu`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=32`
- `--anneal_iter default=20000`
- `--save_iter default=5-10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Model/PLANET_train_esm2_vector_gating.py`** (315 satır):
- `-t`
- `-v`
- `-te`
- `-d`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=1000`
- `--batch_size default=24`
- `--anneal_iter default=50000`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`
- `--binder_per_batch default=2`
- `--decoy_count default=11`

**`Model/PLANET_train_esm2_vector_gating_contrastive_loss.py`** (306 satır):
- `-t`
- `-v`
- `-te`
- `-d`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=32`
- `--binder_per_batch default=16`
- `--decoy_count default=10`
- `--anneal_iter default=20000`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`
- `--num_workers default=12`

**`Model/PLANET_train_esm2_vector_gating_smooth.py`** (296 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512 :: ESM-2 boyutu`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=32`
- `--anneal_iter default=20000`
- `--save_iter default=5-10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Model/PLANET_train_esm2_vector_gating_smooth_contrastive_loss.py`** (296 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512 :: ESM-2 boyutu`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=32`
- `--anneal_iter default=20000`
- `--save_iter default=5-10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Model/PLANET_train_noESM_noUW.py`** (312 satır):
- `-t :: Yol Train.pkl dosyasına.`
- `-v :: Yol Valid.pkl dosyasına.`
- `-te :: Yol Core.pkl dosyasına.`
- `-d :: Model ağırlıklarının kaydedileceği dizin.`
- `-s default=300`
- `-n default=8`
- `-k default=300`
- `-va default=300`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=16`
- `--anneal_iter default=50000`
- `--initial_step default=0`
- `--save_iter default=10000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--binder_per_batch default=2`

**`Scripts/DUDE_benchmark_centered.py`** (359 satır):
- `--model_type`
- `--lmdb_dir`
- `--model_path`
- `--out_dir`
- `--feature_dims default=512`
- `--nheads default=8`
- `--key_dims default=128`
- `--value_dims default=128`
- `--pro_update_inters default=3`
- `--lig_update_iters default=10`
- `--pro_lig_update_iters default=1`
- `--batch_size default=64`
- `--device default=''`
- `--center_ligands`
- `--wandb_project default='PLANET_DUDE'`
- `--run_name default=''`

**`Scripts/LITPCBA_benchmark_centered.py`** (433 satır):
- `--lmdb_dir`
- `--model_type`
- `--model_path`
- `--checkpoint_dir`
- `--out_dir`
- `--feature_dims default=512`
- `--nheads default=8`
- `--key_dims default=128`
- `--value_dims default=128`
- `--pro_update_inters default=3`
- `--lig_update_iters default=10`
- `--pro_lig_update_iters default=1`
- `--batch_size default=128`
- `--device default=''`
- `--center_ligands`
- `--wandb_project default='PLANET_LITPCBA'`
- `--wandb_name default=''`

**`Scripts/PLANET_finetune_crossdocked.py`** (355 satır):
- `--train`
- `--valid :: PDBbind valid LMDB`
- `--test :: PDBbind core LMDB`
- `--save_dir`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-5`
- `--clip_norm default=1.0`
- `--epoch default=50`
- `--batch_size default=12`
- `--decoy_count default=10`
- `--save_iter default=5000`
- `--print_iter default=100`
- `--load_checkpoint :: ESM_VG 1.9M checkpoint yolu`
- `--initial_step default=1900000`
- `--freeze_affinity :: Affinity loss kapalı — sadece contact map öğren`
- `--run_name default=None`

**`Scripts/PLANET_re-evaluate_all.py`** (143 satır):
- `--ckpt_dir`
- `--valid_data`
- `--test_data`
- `--project default="PLANET_EVOLUTION"`
- `--run_name`
- `--dims default=512`

**`Scripts/PLANET_re-evaluate_contrastive.py`** (143 satır):
- `--ckpt_dir`
- `--valid_data`
- `--test_data`
- `--project default="PLANET_EVOLUTION"`
- `--run_name`
- `--dims default=512`

**`Scripts/PLANET_re-evaluate_esm.py`** (143 satır):
- `--ckpt_dir`
- `--valid_data`
- `--test_data`
- `--project default="PLANET_EVOLUTION"`
- `--run_name`
- `--dims default=512`

**`Scripts/PLANET_re-evaluate_noesm.py`** (143 satır):
- `--ckpt_dir`
- `--valid_data`
- `--test_data`
- `--project default="PLANET_EVOLUTION"`
- `--run_name`
- `--dims default=512`

**`Scripts/PLANET_re-evaluate_vector_gating.py`** (143 satır):
- `--ckpt_dir`
- `--valid_data`
- `--test_data`
- `--project default="PLANET_EVOLUTION"`
- `--run_name`
- `--dims default=512`

**`Scripts/PLANET_train_Noesm2_UW.py`** (355 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2.py`** (316 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2_noUW.py`** (330 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2_vector_gating.py`** (336 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2_vector_gating_contrastive_loss.py`** (331 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2_vector_gating_contrastive_loss_mean.py`** (328 satır):
- `-t`
- `-v`
- `-te`
- `-d`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1`
- `--decoy_count default=10`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2_vector_gating_contrastive_loss_v2.py`** (328 satır):
- `-t`
- `-v`
- `-te`
- `-d`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1`
- `--decoy_count default=10`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_esm2_vector_gating_contrastive_loss_v4.py`** (330 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/PLANET_train_noESM_noUW.py`** (327 satır):
- `-t`
- `-v`
- `-te`
- `-d :: Model kayıt dizini`
- `-s default=512`
- `-n default=8`
- `-k default=128`
- `-va default=128`
- `-pu default=3`
- `-lu default=10`
- `-pl default=1`
- `--lr default=1e-4`
- `--clip_norm default=1.0`
- `--epoch default=250`
- `--batch_size default=12`
- `--binder_per_batch default=1 :: Her batch'te kaç pocket: toplam molekül = binder_per_batch *`
- `--decoy_count default=10 :: Valid/test'te her binder için max kaç decoy alınsın`
- `--anneal_iter default=20000`
- `--save_iter default=20000`
- `--print_iter default=1000`
- `--load_checkpoint default=None`
- `--initial_step default=0`

**`Scripts/eval_unified.py`** (245 satır):
- `--model_type :: Model mimarisi: noesm \| esm \| vg \| contrastive`
- `--ckpt_dir`
- `--valid_data`
- `--test_data`
- `--project default='PLANET_THRESHOLD'`
- `--run_name`
- `--dims default=512`
- `--min_step default=0`

**`Scripts/prepare_benchmark_proteins.py`** (182 satır):
- `--benchmark`
- `--json_path`
- `--njobs default=20`

---

# 6. (Otomatik) Model Varyantları — Satır Farkları

```diff

==========================================================================================
GROUP: Model/ ARCHITECTURE (base=PLANET_model.py)
==========================================================================================
BASE Model/PLANET_model.py: 128 satir

--- PLANET_model_esm2.py : 128 satir | +1 -1 (base'e gore)
   @@ -4 +4 @@
   -from layers import ProteinEGNN,LigandGAT,ProLig
   +from layers_esm2 import ProteinEGNN,LigandGAT,ProLig

--- PLANET_model_esm2_gating.py : 128 satir | +1 -1 (base'e gore)
   @@ -4 +4 @@
   -from layers import ProteinEGNN,LigandGAT,ProLig
   +from layers_esm2_gating import ProteinEGNN,LigandGAT,ProLig

--- PLANET_model_esm2_vector.py : 201 satir | +156 -83 (base'e gore)
   @@ -1 +1,4 @@
   -import math,os,sys,torch
   +import math, os, sys, torch
   +import torch.nn as nn
   +
   +# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
   @@ -3,2 +6,2 @@
   -import torch.nn as nn
   -from layers import ProteinEGNN,LigandGAT,ProLig
   +
   +from layers_esm2_vector import ProteinEGNN, LigandGAT, ProLig
   @@ -8 +11 @@
   -    def __init__(self,feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,device):
   +    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
   @@ -12,6 +15,10 @@
   -        self.proteinegnn = ProteinEGNN(feature_dims,pro_update_inters,self.device)
   -        self.ligandgat = LigandGAT(feature_dims,nheads,key_dims,value_dims,lig_update_iters,self.device)
   -        self.prolig = ProLig(feature_dims,nheads,pro_lig_update_iters,self.device)
   -
   -        # 🎯 KRİTİK DÜZELTME 1: Uncertainty Weighting için öğrenilebilir parametreler
   -        # Bunlar, kayıp fonksiyonlarını dinamik olarak ölçeklemek için kullanılacaktır (log(sigma^2) olarak).
   +        
   +        # 1. Katmanlar
   +        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
   +        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
   +        
   +        # ProLig: Artık geometrik hesaplamalar için koordinatları bekleyen yeni versiyonu çağırıyoruz
   +        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device)
   +
   +        # 2. Uncertainty Weighting (Belirsizlik Ağırlıklandırma) Parametreleri
   +        # Train dosyasında loss'ları dinamik dengelemek için kullanılan öğrenilebilir sigma parametreleri.
   @@ -22 +29,2 @@
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum',)
   +        # 3. Kayıp Fonksiyonları
   +        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
   @@ -26,3 +34,10 @@
   -    ###used for model training 
   -    def forward(self,res_batch,mol_batch):
   -        (fresidues,res_map,res_scope,alpha_coordinates) = res_batch
   +    def forward(self, res_batch, mol_batch):
   +        """
   +        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
   +        
   +        res_batch: Protein verisi (Özellikler, Harita, Scope, Koordinatlar)
   +        mol_batch: Ligand verisi (Atomlar, Bağlar, Graf, Scope, KOORDİNATLAR)
   +        """
   +        # --- A. Protein Verisini Hazırla ---
   +        # alpha_coordinates: Protein backbone koordinatları
   +        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
   @@ -30,3 +45,11 @@
   -        #res_map = create_var(res_map)
   -        alpha_coordinates = create_var(alpha_coordinates)
   -        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch
   +        alpha_coordinates = create_var(alpha_coordinates).to(self.device)
   +
   +        # --- B. Ligand Verisini Hazırla ---
   +        # DİKKAT: Dataset'ten lig_coordinates'in (6. eleman) geldiğinden emin olmalıyız.
   +        # Eğer dataset 5 eleman döndürüyorsa 'PLANET_datautils_esm2.py' dosyasında collate_fn güncellenmelidir.
   +        if len(mol_batch) == 6:
   +            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
   +        else:
   +            # Hata yakalama: Eğer eski dataset kullanılıyorsa uyarı ver.
   +            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi. Lütfen datautils dosyasını kon
   +
   @@ -37,4 +60,9 @@
   -
   -        #distance_weight,res_mask = self.distance_network(fresidues,alpha_coordinates,res_scope,res_map)
   -        fresidues = self.proteinegnn(fresidues,alpha_coordinates,res_scope)
   -        fatoms = self.ligandgat(fatoms,fbonds,agraph,bgraph,lig_scope)
   +        lig_coordinates = create_var(lig_coordinates).to(self.device)
   +
   +        # --- C. Alt Katmanlardan Geçiş ---
   +        
   +        # 1. Protein Embedding (EGNN - Koordinat kullanır)
   +        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
   +        
   +        # 2. Ligand Embedding (GAT - Graph tabanlı)
   +        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   @@ -42,2 +70,7 @@
   -        predicted_lig_interactions,predicted_interactions,predicted_affinities = self.prolig(fresidues,fatoms,res_scope,lig_scope)
   -        return (predicted_lig_interactions,predicted_interactions,predicted_affinities)
   +        # 3. Protein-Ligand Etkileşimi (ProLig)
   +        # KRİTİK NOKTA: Hem protein (alpha) hem ligand koordinatlarını buraya paslıyoruz.
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
   +            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
   +        )
   +        
   +        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
   @@ -45,6 +78,14 @@
   -    def compute_loss(self,predictions,targets,res_batch,mol_batch):
   ... (+169 satir daha)

--- PLANET_model_esm2_vector_gating.py : 201 satir | +156 -83 (base'e gore)
   @@ -1 +1,4 @@
   -import math,os,sys,torch
   +import math, os, sys, torch
   +import torch.nn as nn
   +
   +# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
   @@ -3,2 +6,2 @@
   -import torch.nn as nn
   -from layers import ProteinEGNN,LigandGAT,ProLig
   +
   +from layers_esm2_vector_gating import ProteinEGNN, LigandGAT, ProLig
   @@ -8 +11 @@
   -    def __init__(self,feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,device):
   +    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
   @@ -12,6 +15,10 @@
   -        self.proteinegnn = ProteinEGNN(feature_dims,pro_update_inters,self.device)
   -        self.ligandgat = LigandGAT(feature_dims,nheads,key_dims,value_dims,lig_update_iters,self.device)
   -        self.prolig = ProLig(feature_dims,nheads,pro_lig_update_iters,self.device)
   -
   -        # 🎯 KRİTİK DÜZELTME 1: Uncertainty Weighting için öğrenilebilir parametreler
   -        # Bunlar, kayıp fonksiyonlarını dinamik olarak ölçeklemek için kullanılacaktır (log(sigma^2) olarak).
   +        
   +        # 1. Katmanlar
   +        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
   +        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
   +        
   +        # ProLig: Artık geometrik hesaplamalar için koordinatları bekleyen yeni versiyonu çağırıyoruz
   +        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device)
   +
   +        # 2. Uncertainty Weighting (Belirsizlik Ağırlıklandırma) Parametreleri
   +        # Train dosyasında loss'ları dinamik dengelemek için kullanılan öğrenilebilir sigma parametreleri.
   @@ -22 +29,2 @@
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum',)
   +        # 3. Kayıp Fonksiyonları
   +        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
   @@ -26,3 +34,10 @@
   -    ###used for model training 
   -    def forward(self,res_batch,mol_batch):
   -        (fresidues,res_map,res_scope,alpha_coordinates) = res_batch
   +    def forward(self, res_batch, mol_batch):
   +        """
   +        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
   +        
   +        res_batch: Protein verisi (Özellikler, Harita, Scope, Koordinatlar)
   +        mol_batch: Ligand verisi (Atomlar, Bağlar, Graf, Scope, KOORDİNATLAR)
   +        """
   +        # --- A. Protein Verisini Hazırla ---
   +        # alpha_coordinates: Protein backbone koordinatları
   +        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
   @@ -30,3 +45,11 @@
   -        #res_map = create_var(res_map)
   -        alpha_coordinates = create_var(alpha_coordinates)
   -        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch
   +        alpha_coordinates = create_var(alpha_coordinates).to(self.device)
   +
   +        # --- B. Ligand Verisini Hazırla ---
   +        # DİKKAT: Dataset'ten lig_coordinates'in (6. eleman) geldiğinden emin olmalıyız.
   +        # Eğer dataset 5 eleman döndürüyorsa 'PLANET_datautils_esm2.py' dosyasında collate_fn güncellenmelidir.
   +        if len(mol_batch) == 6:
   +            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
   +        else:
   +            # Hata yakalama: Eğer eski dataset kullanılıyorsa uyarı ver.
   +            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi. Lütfen datautils dosyasını kon
   +
   @@ -37,4 +60,9 @@
   -
   -        #distance_weight,res_mask = self.distance_network(fresidues,alpha_coordinates,res_scope,res_map)
   -        fresidues = self.proteinegnn(fresidues,alpha_coordinates,res_scope)
   -        fatoms = self.ligandgat(fatoms,fbonds,agraph,bgraph,lig_scope)
   +        lig_coordinates = create_var(lig_coordinates).to(self.device)
   +
   +        # --- C. Alt Katmanlardan Geçiş ---
   +        
   +        # 1. Protein Embedding (EGNN - Koordinat kullanır)
   +        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
   +        
   +        # 2. Ligand Embedding (GAT - Graph tabanlı)
   +        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   @@ -42,2 +70,7 @@
   -        predicted_lig_interactions,predicted_interactions,predicted_affinities = self.prolig(fresidues,fatoms,res_scope,lig_scope)
   -        return (predicted_lig_interactions,predicted_interactions,predicted_affinities)
   +        # 3. Protein-Ligand Etkileşimi (ProLig)
   +        # KRİTİK NOKTA: Hem protein (alpha) hem ligand koordinatlarını buraya paslıyoruz.
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
   +            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
   +        )
   +        
   +        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
   @@ -45,6 +78,14 @@
   -    def compute_loss(self,predictions,targets,res_batch,mol_batch):
   ... (+169 satir daha)

--- PLANET_model_esm2_vector_gating_contrastive_loss.py : 156 satir | +121 -93 (base'e gore)
   @@ -1 +1,3 @@
   -import math,os,sys,torch
   +import math, os, sys, torch
   +import torch.nn as nn
   +
   @@ -3,2 +5,2 @@
   -import torch.nn as nn
   -from layers import ProteinEGNN,LigandGAT,ProLig
   +
   +from layers_esm2_vector_gating import ProteinEGNN, LigandGAT, ProLig 
   @@ -8 +10 @@
   -    def __init__(self,feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,device):
   +    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
   @@ -12,3 +14,4 @@
   -        self.proteinegnn = ProteinEGNN(feature_dims,pro_update_inters,self.device)
   -        self.ligandgat = LigandGAT(feature_dims,nheads,key_dims,value_dims,lig_update_iters,self.device)
   -        self.prolig = ProLig(feature_dims,nheads,pro_lig_update_iters,self.device)
   +        
   +        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
   +        self.ligandgat   = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
   +        self.prolig      = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device, dropout_rate=0.2)
   @@ -16,4 +19,4 @@
   -        # 🎯 KRİTİK DÜZELTME 1: Uncertainty Weighting için öğrenilebilir parametreler
   -        # Bunlar, kayıp fonksiyonlarını dinamik olarak ölçeklemek için kullanılacaktır (log(sigma^2) olarak).
   -        self.log_sigma_lig = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_pl = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        # Uncertainty Weighting — 3 parametre
   +        # log_sigma_contrastive KALDIRILDI: contrastive log_sigma_affinity ile paylaşılıyor
   +        self.log_sigma_lig      = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_pl       = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   @@ -22 +25 @@
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum',)
   +        self.ligand_interaction_loss  = nn.BCELoss(reduction='sum')
   @@ -24 +27 @@
   -        self.mse_loss = nn.MSELoss(reduction='none')
   +        self.mse_loss                 = nn.MSELoss(reduction='none')
   @@ -26,11 +29,4 @@
   -    ###used for model training 
   -    def forward(self,res_batch,mol_batch):
   -        (fresidues,res_map,res_scope,alpha_coordinates) = res_batch
   -        fresidues = fresidues.to(self.device)
   -        #res_map = create_var(res_map)
   -        alpha_coordinates = create_var(alpha_coordinates)
   -        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch
   -        fatoms = fatoms.to(self.device)
   -        fbonds = fbonds.to(self.device)
   -        agraph = agraph.to(self.device)
   -        bgraph = bgraph.to(self.device)
   +    def forward(self, res_batch, mol_batch):
   +        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
   +        fresidues         = fresidues.to(self.device)
   +        alpha_coordinates = create_var(alpha_coordinates).to(self.device)
   @@ -38,6 +34,18 @@
   -        #distance_weight,res_mask = self.distance_network(fresidues,alpha_coordinates,res_scope,res_map)
   -        fresidues = self.proteinegnn(fresidues,alpha_coordinates,res_scope)
   -        fatoms = self.ligandgat(fatoms,fbonds,agraph,bgraph,lig_scope)
   -         
   -        predicted_lig_interactions,predicted_interactions,predicted_affinities = self.prolig(fresidues,fatoms,res_scope,lig_scope)
   -        return (predicted_lig_interactions,predicted_interactions,predicted_affinities)
   +        if len(mol_batch) == 6:
   +            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
   +        else:
   +            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor, ancak {len(mol_batch)} eleman geldi.")
   +
   +        fatoms          = fatoms.to(self.device)
   +        fbonds          = fbonds.to(self.device)
   +        agraph          = agraph.to(self.device)
   +        bgraph          = bgraph.to(self.device)
   +        lig_coordinates = create_var(lig_coordinates).to(self.device)
   +
   +        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
   +        fatoms    = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   +
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
   +            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
   +        )
   +        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
   @@ -45,4 +53,5 @@
   -    def compute_loss(self,predictions,targets,res_batch,mol_batch):
   -        predicted_lig_interactions,predicted_interactions,predicted_affinities = predictions
   -        (_,_,res_scope,_) = res_batch
   -        (_,_,_,_,lig_scope) = mol_batch
   +    def compute_loss(self, predictions, targets, res_batch, mol_batch):
   +        # compute_loss DEĞİŞMEDİ — hâlâ 4 değer döndürüyor
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions
   +        (_, _, res_scope, _) = res_batch
   +        lig_scope = mol_batch[4]
   @@ -50,2 +59,2 @@
   -        mol_intearctions,pro_lig_interactions,pKs,pK_flags,complex_labels = targets
   -        mol_intearctions = create_var(mol_intearctions)
   ... (+150 satir daha)

--- PLANET_model_esm2_vector_gating_smooth_contrastive_loss.py : 222 satir | +179 -85 (base'e gore)
   @@ -1 +1,4 @@
   -import math,os,sys,torch
   +import math, os, sys, torch
   +import torch.nn as nn
   +
   +# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
   @@ -3,2 +6,2 @@
   -import torch.nn as nn
   -from layers import ProteinEGNN,LigandGAT,ProLig
   +
   +from layers_esm2_vector_gating import ProteinEGNN, LigandGAT, ProLig
   @@ -8 +11 @@
   -    def __init__(self,feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,device):
   +    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
   @@ -12,6 +15,10 @@
   -        self.proteinegnn = ProteinEGNN(feature_dims,pro_update_inters,self.device)
   -        self.ligandgat = LigandGAT(feature_dims,nheads,key_dims,value_dims,lig_update_iters,self.device)
   -        self.prolig = ProLig(feature_dims,nheads,pro_lig_update_iters,self.device)
   -
   -        # 🎯 KRİTİK DÜZELTME 1: Uncertainty Weighting için öğrenilebilir parametreler
   -        # Bunlar, kayıp fonksiyonlarını dinamik olarak ölçeklemek için kullanılacaktır (log(sigma^2) olarak).
   +        
   +        # 1. Katmanlar
   +        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
   +        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
   +        
   +        # ProLig: Artık geometrik hesaplamalar için koordinatları bekleyen yeni versiyonu çağırıyoruz
   +        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device)
   +
   +        # 2. Uncertainty Weighting (Belirsizlik Ağırlıklandırma) Parametreleri
   +        # Train dosyasında loss'ları dinamik dengelemek için kullanılan öğrenilebilir sigma parametreleri.
   @@ -22 +29,2 @@
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum',)
   +        # 3. Kayıp Fonksiyonları
   +        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
   @@ -26,3 +34,10 @@
   -    ###used for model training 
   -    def forward(self,res_batch,mol_batch):
   -        (fresidues,res_map,res_scope,alpha_coordinates) = res_batch
   +    def forward(self, res_batch, mol_batch):
   +        """
   +        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
   +        
   +        res_batch: Protein verisi (Özellikler, Harita, Scope, Koordinatlar)
   +        mol_batch: Ligand verisi (Atomlar, Bağlar, Graf, Scope, KOORDİNATLAR)
   +        """
   +        # --- A. Protein Verisini Hazırla ---
   +        # alpha_coordinates: Protein backbone koordinatları
   +        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
   @@ -30,3 +45,11 @@
   -        #res_map = create_var(res_map)
   -        alpha_coordinates = create_var(alpha_coordinates)
   -        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch
   +        alpha_coordinates = create_var(alpha_coordinates).to(self.device)
   +
   +        # --- B. Ligand Verisini Hazırla ---
   +        # DİKKAT: Dataset'ten lig_coordinates'in (6. eleman) geldiğinden emin olmalıyız.
   +        # Eğer dataset 5 eleman döndürüyorsa 'PLANET_datautils_esm2.py' dosyasında collate_fn güncellenmelidir.
   +        if len(mol_batch) == 6:
   +            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
   +        else:
   +            # Hata yakalama: Eğer eski dataset kullanılıyorsa uyarı ver.
   +            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi. Lütfen datautils dosyasını kon
   +
   @@ -37,4 +60,9 @@
   -
   -        #distance_weight,res_mask = self.distance_network(fresidues,alpha_coordinates,res_scope,res_map)
   -        fresidues = self.proteinegnn(fresidues,alpha_coordinates,res_scope)
   -        fatoms = self.ligandgat(fatoms,fbonds,agraph,bgraph,lig_scope)
   +        lig_coordinates = create_var(lig_coordinates).to(self.device)
   +
   +        # --- C. Alt Katmanlardan Geçiş ---
   +        
   +        # 1. Protein Embedding (EGNN - Koordinat kullanır)
   +        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
   +        
   +        # 2. Ligand Embedding (GAT - Graph tabanlı)
   +        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   @@ -42,2 +70,7 @@
   -        predicted_lig_interactions,predicted_interactions,predicted_affinities = self.prolig(fresidues,fatoms,res_scope,lig_scope)
   -        return (predicted_lig_interactions,predicted_interactions,predicted_affinities)
   +        # 3. Protein-Ligand Etkileşimi (ProLig)
   +        # KRİTİK NOKTA: Hem protein (alpha) hem ligand koordinatlarını buraya paslıyoruz.
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
   +            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
   +        )
   +        
   +        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
   @@ -45,6 +78,7 @@
   -    def compute_loss(self,predictions,targets,res_batch,mol_batch):
   ... (+193 satir daha)

==========================================================================================
GROUP: Model/ LAYERS (base=layers.py)
==========================================================================================
BASE Model/layers.py: 392 satir

--- layers_esm2.py : 392 satir | +2 -2 (base'e gore)
   @@ -18 +18 @@
   -        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
   +        self.res_embedding = nn.Sequential(nn.Linear(512,feature_dims,bias=False),nn.LeakyReLU(0.1))
   @@ -261 +261 @@
   -        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
   +        self.res_embedding = nn.Sequential(nn.Linear(512,feature_dims,bias=False),nn.LeakyReLU(0.1))

--- layers_esm2_gating.py : 366 satir | +220 -246 (base'e gore)
   @@ -1,2 +0,0 @@
   -from asyncio import coroutines
   -from venv import create
   @@ -6,4 +4,4 @@
   -import math,sys
   -from nnutils import create_var
   -from nnutils import index_select_ND
   -
   +import math
   +from nnutils import create_var, index_select_ND
   +
   +# --- ProteinGAT (DEĞİŞİKLİK YOK) ---
   @@ -18 +16 @@
   -        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
   +        self.res_embedding = nn.Sequential(nn.Linear(512,feature_dims,bias=False),nn.LeakyReLU(0.1))
   @@ -26,8 +23,0 @@
   -        #self.super_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
   -        #self.res_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
   -        #self.linear_att = nn.Linear(feature_dims,1)
   -        #self.aggregate = nn.Sequential(nn.Linear(nheads*feature_dims,feature_dims),nn.LeakyReLU(0.1))
   -
   -        #para group
   -        #self.interaction_modules = [self.res_embedding,self.linear_Q,self.linear_K,self.linear_V,self.linear_out]
   -
   @@ -35,3 +25,3 @@
   -        q = torch.reshape(q,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,0,2]) #[h,n,k]
   -        k = torch.reshape(k,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,2,0]) #[h,k,n]
   -        v = torch.reshape(v,shape=[self.node_size,self.nheads,self.value_dims]).permute([1,0,2]) #[h,n,v]
   +        q = torch.reshape(q,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,0,2]) 
   +        k = torch.reshape(k,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,2,0]) 
   +        v = torch.reshape(v,shape=[self.node_size,self.nheads,self.value_dims]).permute([1,0,2]) 
   @@ -46 +36 @@
   -            q = self.linear_Q(fresidues)  #[batch,nodes,nheads*key]
   +            q = self.linear_Q(fresidues)
   @@ -49 +38,0 @@
   -            #v = fresidues.unsqueeze(0)   #[1,n,f]
   @@ -52,6 +41,4 @@
   -            attention_matrix = attention_matrix + res_mask #[h,n,n]
   -            attention_matrix = torch.softmax(attention_matrix,dim=-1) #  * distance_weight #[h,n,n]
   -            #attention_matrix = (attention_matrix + torch.transpose(attention_matrix,-2,-1)) / 2. #[h,n,n]
   -            #attention_matrix = attention_matrix / torch.sum(attention_matrix,dim=-1,keepdim=True)  #normalize to 1
   -            v = torch.matmul(attention_matrix,v).permute([1,0,2]) #[h,n,f] -- [n,h,f]
   -            fresidues = self.linear_out(torch.sum(v,dim=1)) #[n,f]
   +            attention_matrix = attention_matrix + res_mask
   +            attention_matrix = torch.softmax(attention_matrix,dim=-1) 
   +            v = torch.matmul(attention_matrix,v).permute([1,0,2])
   +            fresidues = self.linear_out(torch.sum(v,dim=1))
   @@ -60,0 +48 @@
   +# --- LigandGAT (DEĞİŞİKLİK YOK) ---
   @@ -77,3 +64,0 @@
   -        #para group
   -        #self.interaction_modules = [self.fbonds_embedding,self.linear_Q,self.linear_K,self.linear_V,self.fbonds_out,self.linear_last,self.linear_lig_interacti
   -    
   @@ -82,3 +67,3 @@
   -        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3])  #[b,1,h,k] -- #[b,h,1,k]
   -        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])  #[b,h,k,nei]
   -        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])  #[b,h,nei,v]
   +        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3]) 
   +        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])
   +        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])
   @@ -94 +79 @@
   -        fbonds_input = self.fbonds_embedding(fbonds) #[fbonds,feature_dim]
   +        fbonds_input = self.fbonds_embedding(fbonds)
   @@ -97 +82 @@
   -        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)  #[fbonds,1,1,nei]
   +        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)
   @@ -99 +84 @@
   -            nei_message = index_select_ND(message,0,bgraph)  #[fbonds,nei,feature_dim]
   +            nei_message = index_select_ND(message,0,bgraph)
   @@ -103 +87,0 @@
   -            #v = nei_message.reshape([self.bonds_size,1,self.nei_size,self.feature_dims])
   @@ -105 +89 @@
   -            attention_matrix = torch.matmul(q,k)/ math.sqrt(self.key_dims) + nei_mask   #[b,h,1,nei]
   +            attention_matrix = torch.matmul(q,k)/ math.sqrt(self.key_dims) + nei_mask
   @@ -107,2 +91 @@
   -            #print(attention_matrix)
   -            nei_message = torch.matmul(attention_matrix,v).permute([0,2,1,3])  #[b,h,1,v] -- [b,1,h,v]
   +            nei_message = torch.matmul(attention_matrix,v).permute([0,2,1,3])
   @@ -116,65 +99,23 @@
   -        
   -class ProLig(nn.Module):
   -    def __init__(self,feature_dims,nheads,update_iters,device):
   -        super().__init__()
   -        self.feature_dims = feature_dims
   -        self.device = device
   -        self.prolig_attention = ProteinLigandAttention(feature_dims,nheads,update_iters,device)
   -
   -        #ligand interaction prediction weight
   -        self.linear_lig_interaction = nn.Sequential(
   -            nn.Linear(feature_dims,feature_dims,bias=False),
   ... (+412 satir daha)

--- layers_esm2_vector.py : 374 satir | +145 -163 (base'e gore)
   @@ -1,2 +0,0 @@
   -from asyncio import coroutines
   -from venv import create
   @@ -6 +4 @@
   -import math,sys
   +import math, sys
   @@ -9,0 +8 @@
   +# --- DEĞİŞİKLİK YOK (ProteinGAT) ---
   @@ -18 +17 @@
   -        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
   +        self.res_embedding = nn.Sequential(nn.Linear(512,feature_dims,bias=False),nn.LeakyReLU(0.1))
   @@ -25,8 +23,0 @@
   -
   -        #self.super_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
   -        #self.res_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
   -        #self.linear_att = nn.Linear(feature_dims,1)
   -        #self.aggregate = nn.Sequential(nn.Linear(nheads*feature_dims,feature_dims),nn.LeakyReLU(0.1))
   -
   -        #para group
   -        #self.interaction_modules = [self.res_embedding,self.linear_Q,self.linear_K,self.linear_V,self.linear_out]
   @@ -49 +39,0 @@
   -            #v = fresidues.unsqueeze(0)   #[1,n,f]
   @@ -53,3 +43 @@
   -            attention_matrix = torch.softmax(attention_matrix,dim=-1) #  * distance_weight #[h,n,n]
   -            #attention_matrix = (attention_matrix + torch.transpose(attention_matrix,-2,-1)) / 2. #[h,n,n]
   -            #attention_matrix = attention_matrix / torch.sum(attention_matrix,dim=-1,keepdim=True)  #normalize to 1
   +            attention_matrix = torch.softmax(attention_matrix,dim=-1) 
   @@ -60,0 +49 @@
   +# --- DEĞİŞİKLİK YOK (LigandGAT) ---
   @@ -77,3 +65,0 @@
   -        #para group
   -        #self.interaction_modules = [self.fbonds_embedding,self.linear_Q,self.linear_K,self.linear_V,self.fbonds_out,self.linear_last,self.linear_lig_interacti
   -    
   @@ -82,3 +68,3 @@
   -        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3])  #[b,1,h,k] -- #[b,h,1,k]
   -        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])  #[b,h,k,nei]
   -        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])  #[b,h,nei,v]
   +        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3]) 
   +        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])
   +        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])
   @@ -97 +83 @@
   -        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)  #[fbonds,1,1,nei]
   +        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)
   @@ -103 +88,0 @@
   -            #v = nei_message.reshape([self.bonds_size,1,self.nei_size,self.feature_dims])
   @@ -107 +91,0 @@
   -            #print(attention_matrix)
   @@ -116 +100,108 @@
   -        
   +
   +# --- GaussianSmearing (Vector için gerekli) ---
   +class GaussianSmearing(nn.Module):
   +    def __init__(self, start=0.0, stop=20.0, num_gaussians=32):
   +        super().__init__()
   +        offset = torch.linspace(start, stop, num_gaussians)
   +        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2
   +        self.register_buffer('offset', offset)
   +
   +    def forward(self, dist):
   +        dist = dist.unsqueeze(-1) - self.offset
   +        return torch.exp(self.coeff * torch.pow(dist, 2))
   +
   +# --- [GÜNCELLENDİ] ProteinLigandAttention (+Dropout) ---
   +class ProteinLigandAttention(nn.Module):
   +    def __init__(self, feature_dims, nheads, update_iters, device, dropout_rate=0.2): # Dropout parametresi eklendi
   +        super().__init__()
   +        self.feature_dims = feature_dims
   +        self.nheads = nheads
   +        self.update_iters = update_iters
   +        self.device = device
   +        
   +        # Standart dönüşümler
   +        self.lig_transform = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +        self.res_transform = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +
   +        self.lig_Q = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.lig_K = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.lig_V = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.lig_out = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +
   +        self.res_Q = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.res_K = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.res_V = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.res_out = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +
   +        # --- Geometrik Modüller ---
   +        self.num_rbf = 32
   +        self.rbf_fn = GaussianSmearing(start=0.0, stop=15.0, num_gaussians=self.num_rbf)
   +        
   +        # [GÜNCELLEME] Geometrik MLP'ye Dropout eklendi
   ... (+251 satir daha)

--- layers_esm2_vector_gating.py : 426 satir | +233 -199 (base'e gore)
   @@ -1,2 +0,0 @@
   -from asyncio import coroutines
   -from venv import create
   @@ -6 +4 @@
   -import math,sys
   +import math, sys
   @@ -9,0 +8 @@
   +# --- DEĞİŞİKLİK YOK (ProteinGAT) ---
   @@ -18 +17 @@
   -        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
   +        self.res_embedding = nn.Sequential(nn.Linear(512,feature_dims,bias=False),nn.LeakyReLU(0.1))
   @@ -25,8 +23,0 @@
   -
   -        #self.super_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
   -        #self.res_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
   -        #self.linear_att = nn.Linear(feature_dims,1)
   -        #self.aggregate = nn.Sequential(nn.Linear(nheads*feature_dims,feature_dims),nn.LeakyReLU(0.1))
   -
   -        #para group
   -        #self.interaction_modules = [self.res_embedding,self.linear_Q,self.linear_K,self.linear_V,self.linear_out]
   @@ -49 +39,0 @@
   -            #v = fresidues.unsqueeze(0)   #[1,n,f]
   @@ -53,3 +43 @@
   -            attention_matrix = torch.softmax(attention_matrix,dim=-1) #  * distance_weight #[h,n,n]
   -            #attention_matrix = (attention_matrix + torch.transpose(attention_matrix,-2,-1)) / 2. #[h,n,n]
   -            #attention_matrix = attention_matrix / torch.sum(attention_matrix,dim=-1,keepdim=True)  #normalize to 1
   +            attention_matrix = torch.softmax(attention_matrix,dim=-1) 
   @@ -60,0 +49 @@
   +# --- DEĞİŞİKLİK YOK (LigandGAT) ---
   @@ -77,3 +65,0 @@
   -        #para group
   -        #self.interaction_modules = [self.fbonds_embedding,self.linear_Q,self.linear_K,self.linear_V,self.fbonds_out,self.linear_last,self.linear_lig_interacti
   -    
   @@ -82,3 +68,3 @@
   -        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3])  #[b,1,h,k] -- #[b,h,1,k]
   -        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])  #[b,h,k,nei]
   -        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])  #[b,h,nei,v]
   +        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3]) 
   +        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])
   +        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])
   @@ -97 +83 @@
   -        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)  #[fbonds,1,1,nei]
   +        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)
   @@ -103 +88,0 @@
   -            #v = nei_message.reshape([self.bonds_size,1,self.nei_size,self.feature_dims])
   @@ -107 +91,0 @@
   -            #print(attention_matrix)
   @@ -116 +100,108 @@
   -        
   +
   +# --- GaussianSmearing (Vector için gerekli) ---
   +class GaussianSmearing(nn.Module):
   +    def __init__(self, start=0.0, stop=20.0, num_gaussians=32):
   +        super().__init__()
   +        offset = torch.linspace(start, stop, num_gaussians)
   +        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2
   +        self.register_buffer('offset', offset)
   +
   +    def forward(self, dist):
   +        dist = dist.unsqueeze(-1) - self.offset
   +        return torch.exp(self.coeff * torch.pow(dist, 2))
   +
   +# --- [GÜNCELLENDİ] ProteinLigandAttention (+Dropout) ---
   +class ProteinLigandAttention(nn.Module):
   +    def __init__(self, feature_dims, nheads, update_iters, device, dropout_rate=0.2): # Dropout parametresi eklendi
   +        super().__init__()
   +        self.feature_dims = feature_dims
   +        self.nheads = nheads
   +        self.update_iters = update_iters
   +        self.device = device
   +        
   +        # Standart dönüşümler
   +        self.lig_transform = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +        self.res_transform = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +
   +        self.lig_Q = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.lig_K = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.lig_V = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.lig_out = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +
   +        self.res_Q = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.res_K = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.res_V = nn.Linear(feature_dims, feature_dims, bias=False)
   +        self.res_out = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
   +
   +        # --- Geometrik Modüller ---
   +        self.num_rbf = 32
   +        self.rbf_fn = GaussianSmearing(start=0.0, stop=15.0, num_gaussians=self.num_rbf)
   +        
   +        # [GÜNCELLEME] Geometrik MLP'ye Dropout eklendi
   ... (+385 satir daha)

==========================================================================================
GROUP: Model/ TRAIN (base=PLANET_train.py)
==========================================================================================
BASE Model/PLANET_train.py: 293 satir

--- PLANET_train2.py : 202 satir | +140 -231 (base'e gore)
   @@ -4,2 +4,4 @@
   -from torch.utils.data import DataLoader
   -import argparse, os, sys
   +import torch.optim.lr_scheduler as lr_scheduler
   +from torch.utils.data import DataLoader, Dataset
   +from torch.utils.data.dataloader import default_collate
   +import argparse, os, sys, pickle
   @@ -6,0 +9 @@
   +import pandas as pd
   @@ -9,3 +12,6 @@
   -from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
   -
   -# --- İMPORT AYARLARI ---
   +# 🎯 YENİ IMPORT: precision_recall_curve eklendi
   +from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
   +                             precision_score, recall_score, mean_squared_error, 
   +                             precision_recall_curve)
   +
   +# --- PYTHON PATH GÜNCELLEMESİ ---
   @@ -16,0 +23 @@
   +# --- KRİTİK IMPORT'LAR ---
   @@ -20,2 +27 @@
   -    
   -    # RDKit uyarılarını kapat
   +    from chemutils import ComplexPocket,tensorize_all,role_of_5
   @@ -27 +32,0 @@
   -# --- YARDIMCI FONKSİYONLAR ---
   @@ -29,5 +34,2 @@
   -    """Veri yapısı içindeki tüm Tensor'ları hedef cihaza taşır."""
   -    if isinstance(data, torch.Tensor):
   -        return data.to(device)
   -    if isinstance(data, (list, tuple)):
   -        return [move_to_device(t, device) for t in data]
   +    if isinstance(data, torch.Tensor): return data.to(device)
   +    if isinstance(data, (list, tuple)): return [move_to_device(t, device) for t in data]
   @@ -38,7 +40,4 @@
   -    # Dosya Yolları
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   -    parser.add_argument('-d','--save_dir', required=True, help="Model kayıt dizini")
   -
   -    # Model Hiperparametreleri
   +    parser.add_argument('-t','--train', required=True)
   +    parser.add_argument('-v','--valid',required=True)
   +    parser.add_argument('-te','--test',required=True)
   +    parser.add_argument('-d','--save_dir', required=True)
   @@ -50,6 +49,4 @@
   -    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
   -    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)
   -    
   -    # Eğitim Ayarları
   -    parser.add_argument('--lr', type=float, default=1e-4)
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
   +    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
   +    parser.add_argument('--lr', type=float, default=1e-5)
   +    parser.add_argument('--clip_norm', type=float, default=1.0)
   @@ -57 +54 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   +    parser.add_argument('--batch_size',type=int,default=24)
   @@ -59 +56,2 @@
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--initial_step',type=int,default=0)
   +    parser.add_argument('--save_iter', type=int, default=10000)
   @@ -62 +60 @@
   -    parser.add_argument('--initial_step', type=int, default=0)
   +    parser.add_argument('--fine_tune_affinity', action='store_true')
   @@ -67,4 +65 @@
   -    # 1. WANDB BAŞLATMA
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_Training", name=f"run_gpu_batch{args.batch_size}_ep{args.epoch}", config=args)
   @@ -72,14 +67,11 @@
   -    print(f"✅ Eğitim Cihazı: {device}")
   -
   -    # Validasyon & Test (Shuffle False, Decoy True)
   -    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True)
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   -
   -    # 3. MODEL BAŞLATMA
   +    print(f"Eğitim Cihazı: {device}")
   ... (+307 satir daha)

--- PLANET_train_Noesm2_UW.py : 361 satir | +250 -182 (base'e gore)
   @@ -0,0 +1 @@
   +# PLANET_train.py (DÜZENLENMİŞ)
   @@ -4,2 +5,4 @@
   -from torch.utils.data import DataLoader
   -import argparse, os, sys
   +import torch.optim.lr_scheduler as lr_scheduler
   +from torch.utils.data import DataLoader, Dataset
   +from torch.utils.data.dataloader import default_collate
   +import argparse, os, sys, pickle
   @@ -6,0 +10 @@
   +import pandas as pd
   @@ -10,2 +14,5 @@
   -
   -# --- İMPORT AYARLARI ---
   +import matplotlib.pyplot as plt
   +import seaborn as sns
   +from scipy.stats import pearsonr, spearmanr
   +
   +# --- PYTHON PATH GÜNCELLEMESİ (Modül Bulma Hatasını Çözer) ---
   @@ -13,0 +21 @@
   +
   @@ -16 +24,3 @@
   -
   +# -----------------------------------------------------------
   +
   +# --- KRİTİK PROJE İÇİ IMPORT'LAR VE HATA YÖNETİMİ ---
   @@ -20,2 +30,2 @@
   -    
   -    # RDKit uyarılarını kapat
   +    from chemutils import ComplexPocket,tensorize_all,role_of_5
   +    
   @@ -22,0 +33 @@
   +    
   @@ -26,2 +37,3 @@
   -
   -# --- YARDIMCI FONKSİYONLAR ---
   +# ----------------------------------------------------
   +
   +# PyTorch'a özel Tensor taşıma yardımcı fonksiyonu
   @@ -36,0 +49 @@
   +    
   @@ -38,7 +51,7 @@
   -    # Dosya Yolları
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   -    parser.add_argument('-d','--save_dir', required=True, help="Model kayıt dizini")
   -
   -    # Model Hiperparametreleri
   +    # Argümanlar
   +    parser.add_argument('-t','--train', required=True, help="Yol Train.pkl dosyasına.")
   +    parser.add_argument('-v','--valid',required=True, help="Yol Valid.pkl dosyasına.")
   +    parser.add_argument('-te','--test',required=True, help="Yol Core.pkl dosyasına.")
   +    parser.add_argument('-d','--save_dir', required=True, help="Model ağırlıklarının kaydedileceği dizin.")
   +
   +    # Model Parametreleri (Çift Atamadan kaçınıldı)
   @@ -48,0 +62 @@
   +
   @@ -50,4 +64,3 @@
   -    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
   -    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)
   -    
   -    # Eğitim Ayarları
   +    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
   +    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
   +    
   @@ -55 +68,2 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0)
   +
   @@ -57,3 +71,5 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   -    parser.add_argument('--anneal_iter', type=int, default=20000)
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--batch_size',type=int,default=16)
   +    parser.add_argument('--anneal_iter', type=int, default=50000)
   +
   +    parser.add_argument('--initial_step',type=int,default=0)
   +    parser.add_argument('--save_iter', type=int, default=10000)
   @@ -60,0 +77 @@
   +    parser.add_argument('--binder_per_batch', type=int, default=2)
   @@ -62 +79,4 @@
   -    parser.add_argument('--initial_step', type=int, default=0)
   +    # 🎯 YENİ ARGÜMAN: İnce Ayar Modu (Sadece Afinite Kaybı)
   +    parser.add_argument('--fine_tune_affinity', action='store_true', 
   +                        help="Etkinleştirilirse, geri yayılımda sadece Affinity Loss kullanılır.")
   +
   @@ -65,0 +86 @@
   +    print(args)
   @@ -68,3 +89,11 @@
   ... (+396 satir daha)

--- PLANET_train_esm2.py : 330 satir | +157 -120 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle # <--- EKLENDİ (Dosyayı okumak için şart)
   @@ -18,2 +19,2 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   +    from PLANET_datautils_esm2 import ProLigDataset 
   +    from PLANET_model_esm2 import PLANET 
   @@ -39,3 +40,3 @@
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   +    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   +    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   +    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   @@ -44,2 +45,2 @@
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   +    # Model Hiperparametreleri (ESM-2 İÇİN 512 YAPILDI)
   +    parser.add_argument('-s','--feature_dims', type=int, default=512, help="ESM-2 boyutu")
   @@ -47,2 +48,2 @@
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   +    parser.add_argument('-k','--key_dims', type=int, default=128)
   +    parser.add_argument('-va','--value_dims', type=int, default=128)
   @@ -55 +56 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0) 
   @@ -57,3 +58,3 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   -    parser.add_argument('--anneal_iter', type=int, default=20000)
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--batch_size', type=int, default=32)
   +    parser.add_argument('--anneal_iter', type=int, default=50000)
   +    parser.add_argument('--save_iter', type=int, default=5-10000) 
   @@ -62,0 +64 @@
   +    parser.add_argument('--binder_per_batch', type=int, default=2)
   @@ -68,3 +70,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_TRAINING", name=f"esm2_uw{args.batch_size}", config=args)
   +
   +    # 2. CİHAZ
   @@ -74 +76,9 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   +    # 3. VERİ YÜKLEME (BURASI DÜZELTİLDİ)
   +    # ProLigDataset string path değil, liste istiyor. O yüzden önce biz yüklüyoruz.
   +    print("📂 Veri listeleri yükleniyor...")
   +    with open(args.train, 'rb') as f: train_list = pickle.load(f)
   +    with open(args.valid, 'rb') as f: valid_list = pickle.load(f)
   +    with open(args.test, 'rb') as f:  test_list = pickle.load(f)
   +
   +    # --- 🌟 DÜZELTME BURADA 🌟 ---
   +    # valid_list yerine args.valid, test_list yerine args.test gönderiyoruz.
   @@ -76,8 +86,25 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   +    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   +    
   +    train_dataset = ProLigDataset(
   +        args.train, 
   +        args.batch_size, 
   +        shuffle=True, 
   +        decoy_flag=True, 
   +        binder_per_batch = args.binder_per_batch
   +    )
   +
   +    # DataLoaders
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    
   +    train_loader = DataLoader(
   +        train_dataset,
   +        batch_size=1, # ProLigDataset kendi içinde batch'liyor
   +        num_workers=16,
   +        pin_memory=True,
   +        prefetch_factor=4,
   +        shuffle=True, # Veri düzeyinde rastgelelik
   +        collate_fn=lambda x: x[0]
   +    )
   +    
   +    print(f"   Train: {len(train_list)} | Valid: {len(valid_list)} | Test: {len(test_list)}")
   @@ -89 +116 @@
   -    # Parametre Başlatma (Initialization)
   ... (+220 satir daha)

--- PLANET_train_esm2_gating.py : 296 satir | +93 -90 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle # <--- EKLENDİ (Dosyayı okumak için şart)
   @@ -18,2 +19,2 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   +    from PLANET_datautils_esm2 import ProLigDataset 
   +    from PLANET_model_esm2_gating import PLANET 
   @@ -39,3 +40,3 @@
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   +    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   +    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   +    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   @@ -44,2 +45,2 @@
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   +    # Model Hiperparametreleri (ESM-2 İÇİN 512 YAPILDI)
   +    parser.add_argument('-s','--feature_dims', type=int, default=512, help="ESM-2 boyutu")
   @@ -47,2 +48,2 @@
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   +    parser.add_argument('-k','--key_dims', type=int, default=128)
   +    parser.add_argument('-va','--value_dims', type=int, default=128)
   @@ -55 +56 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0) 
   @@ -57 +58 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   +    parser.add_argument('--batch_size', type=int, default=32)
   @@ -59 +60 @@
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--save_iter', type=int, default=5-10000) 
   @@ -68,3 +69,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_ESM2", name=f"ESM_512_BS{args.batch_size}", config=args)
   +
   +    # 2. CİHAZ
   @@ -74 +75,9 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   +    # 3. VERİ YÜKLEME (BURASI DÜZELTİLDİ)
   +    # ProLigDataset string path değil, liste istiyor. O yüzden önce biz yüklüyoruz.
   +    print("📂 Veri listeleri yükleniyor...")
   +    with open(args.train, 'rb') as f: train_list = pickle.load(f)
   +    with open(args.valid, 'rb') as f: valid_list = pickle.load(f)
   +    with open(args.test, 'rb') as f:  test_list = pickle.load(f)
   +
   +    # --- 🌟 DÜZELTME BURADA 🌟 ---
   +    # valid_list yerine args.valid, test_list yerine args.test gönderiyoruz.
   @@ -76,8 +85,25 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   +    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   +    
   +    train_dataset = ProLigDataset(
   +        args.train, 
   +        args.batch_size, 
   +        shuffle=True, 
   +        decoy_flag=True, 
   +        binder_per_batch=2 
   +    )
   +
   +    # DataLoaders
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    
   +    train_loader = DataLoader(
   +        train_dataset,
   +        batch_size=1, # ProLigDataset kendi içinde batch'liyor
   +        num_workers=20,
   +        pin_memory=True,
   +        prefetch_factor=4,
   +        shuffle=True, # Veri düzeyinde rastgelelik
   +        collate_fn=lambda x: x[0]
   +    )
   +    
   +    print(f"   Train: {len(train_list)} | Valid: {len(valid_list)} | Test: {len(test_list)}")
   @@ -89 +115 @@
   -    # Parametre Başlatma (Initialization)
   +    # Parametre Başlatma
   @@ -124 +150 @@
   -                print(f"⚠️ Veri Yükleme Hatası (Step {total_step}): {e}", file=sys.stderr)
   ... (+124 satir daha)

--- PLANET_train_esm2_noUW.py : 322 satir | +169 -140 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle
   @@ -9 +9,0 @@
   -from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
   @@ -18,2 +18,2 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   +    from PLANET_datautils_esm2 import ProLigDataset 
   +    from PLANET_model_esm2 import PLANET 
   @@ -39,3 +39,3 @@
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   +    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   +    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   +    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   @@ -45 +45 @@
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   +    parser.add_argument('-s','--feature_dims', type=int, default=512, help="ESM-2 boyutu")
   @@ -47,2 +47,2 @@
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   +    parser.add_argument('-k','--key_dims', type=int, default=128)
   +    parser.add_argument('-va','--value_dims', type=int, default=128)
   @@ -55 +55 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0) 
   @@ -57,3 +57,3 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   -    parser.add_argument('--anneal_iter', type=int, default=20000)
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--batch_size', type=int, default=24)
   +    parser.add_argument('--anneal_iter', type=int, default=50000)
   +    parser.add_argument('--save_iter', type=int, default=10000) 
   @@ -62,0 +63 @@
   +    parser.add_argument('--binder_per_batch', type=int, default=2)
   @@ -68,3 +69,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_TRAINING", name=f"esm2_noUW{args.batch_size}", config=args)
   +
   +    # 2. CİHAZ
   @@ -73,2 +74,9 @@
   -
   -    # Validasyon & Test (Shuffle False, Decoy True)
   +    
   +    # 3. VERİ YÜKLEME
   +    print("📂 Veri listeleri yükleniyor...")
   +    with open(args.train, 'rb') as f: train_list = pickle.load(f)
   +    with open(args.valid, 'rb') as f: valid_list = pickle.load(f)
   +    with open(args.test, 'rb') as f:  test_list = pickle.load(f)
   +
   +    # --- 🌟 DÜZELTME BURADA 🌟 ---
   +    # valid_list yerine args.valid, test_list yerine args.test gönderiyoruz.
   @@ -76,8 +84,25 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   +    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   +    
   +    train_dataset = ProLigDataset(
   +        args.train, 
   +        args.batch_size, 
   +        shuffle=True, 
   +        decoy_flag=True, 
   +        binder_per_batch = args.binder_per_batch
   +    )
   +
   +    # DataLoaders
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    
   +    train_loader = DataLoader(
   +        train_dataset,
   +        batch_size=1, # ProLigDataset kendi içinde batch'liyor
   +        num_workers=16,
   +        pin_memory=True,
   +        prefetch_factor=4,
   +        shuffle=True, # Veri düzeyinde rastgelelik
   +        collate_fn=lambda x: x[0]
   +    )
   +
   +    print(f"   Train: {len(train_list)} | Valid: {len(valid_list)} | Test: {len(test_list)}")
   @@ -89 +114 @@
   ... (+251 satir daha)

--- PLANET_train_esm2_vector.py : 296 satir | +93 -90 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle # <--- EKLENDİ (Dosyayı okumak için şart)
   @@ -18,2 +19,2 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   +    from PLANET_datautils_esm2_vector import ProLigDataset 
   +    from PLANET_model_esm2_vector import PLANET 
   @@ -39,3 +40,3 @@
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   +    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   +    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   +    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   @@ -44,2 +45,2 @@
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   +    # Model Hiperparametreleri (ESM-2 İÇİN 512 YAPILDI)
   +    parser.add_argument('-s','--feature_dims', type=int, default=512, help="ESM-2 boyutu")
   @@ -47,2 +48,2 @@
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   +    parser.add_argument('-k','--key_dims', type=int, default=128)
   +    parser.add_argument('-va','--value_dims', type=int, default=128)
   @@ -55 +56 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0) 
   @@ -57 +58 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   +    parser.add_argument('--batch_size', type=int, default=32)
   @@ -59 +60 @@
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--save_iter', type=int, default=5-10000) 
   @@ -68,3 +69,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_ESM2", name=f"ESM_512_BS{args.batch_size}", config=args)
   +
   +    # 2. CİHAZ
   @@ -74 +75,9 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   +    # 3. VERİ YÜKLEME (BURASI DÜZELTİLDİ)
   +    # ProLigDataset string path değil, liste istiyor. O yüzden önce biz yüklüyoruz.
   +    print("📂 Veri listeleri yükleniyor...")
   +    with open(args.train, 'rb') as f: train_list = pickle.load(f)
   +    with open(args.valid, 'rb') as f: valid_list = pickle.load(f)
   +    with open(args.test, 'rb') as f:  test_list = pickle.load(f)
   +
   +    # --- 🌟 DÜZELTME BURADA 🌟 ---
   +    # valid_list yerine args.valid, test_list yerine args.test gönderiyoruz.
   @@ -76,8 +85,25 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   +    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   +    
   +    train_dataset = ProLigDataset(
   +        args.train, 
   +        args.batch_size, 
   +        shuffle=True, 
   +        decoy_flag=True, 
   +        binder_per_batch=2 
   +    )
   +
   +    # DataLoaders
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    
   +    train_loader = DataLoader(
   +        train_dataset,
   +        batch_size=1, # ProLigDataset kendi içinde batch'liyor
   +        num_workers=20,
   +        pin_memory=True,
   +        prefetch_factor=4,
   +        shuffle=True, # Veri düzeyinde rastgelelik
   +        collate_fn=lambda x: x[0]
   +    )
   +    
   +    print(f"   Train: {len(train_list)} | Valid: {len(valid_list)} | Test: {len(test_list)}")
   @@ -89 +115 @@
   -    # Parametre Başlatma (Initialization)
   +    # Parametre Başlatma
   @@ -124 +150 @@
   -                print(f"⚠️ Veri Yükleme Hatası (Step {total_step}): {e}", file=sys.stderr)
   ... (+124 satir daha)

--- PLANET_train_esm2_vector_gating.py : 315 satir | +226 -204 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle
   @@ -10,2 +11,2 @@
   -
   -# --- İMPORT AYARLARI ---
   +from scipy.stats import pearsonr, spearmanr
   +
   @@ -18,5 +19,3 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   -    
   -    # RDKit uyarılarını kapat
   -    RDLogger.DisableLog('rdApp.*') 
   +    from PLANET_datautils_esm2_vector import ProLigDataset
   +    from PLANET_model_esm2_vector_gating import PLANET
   +    RDLogger.DisableLog('rdApp.*')
   @@ -27 +25,0 @@
   -# --- YARDIMCI FONKSİYONLAR ---
   @@ -29 +26,0 @@
   -    """Veri yapısı içindeki tüm Tensor'ları hedef cihaza taşır."""
   @@ -38,25 +35,28 @@
   -    # Dosya Yolları
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   -    parser.add_argument('-d','--save_dir', required=True, help="Model kayıt dizini")
   -
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   -    parser.add_argument('-n','--nheads', type=int, default=8)
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   -    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
   -    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
   -    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)
   -    
   -    # Eğitim Ayarları
   -    parser.add_argument('--lr', type=float, default=1e-4)
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   -    parser.add_argument('--epoch', type=int, default=250)
   -    parser.add_argument('--batch_size', type=int, default=16)
   -    parser.add_argument('--anneal_iter', type=int, default=20000)
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   -    parser.add_argument('--print_iter', type=int, default=1000)
   -    parser.add_argument('--load_checkpoint', default=None)
   -    parser.add_argument('--initial_step', type=int, default=0)
   +
   +    # Dosya yolları
   +    parser.add_argument('-t',  '--train',  required=True)
   +    parser.add_argument('-v',  '--valid',  required=True)
   +    parser.add_argument('-te', '--test',   required=True)
   +    parser.add_argument('-d',  '--save_dir', required=True)
   +
   +    # Model
   +    parser.add_argument('-s',  '--feature_dims',         type=int,   default=512)
   +    parser.add_argument('-n',  '--nheads',               type=int,   default=8)
   +    parser.add_argument('-k',  '--key_dims',             type=int,   default=128)
   +    parser.add_argument('-va', '--value_dims',           type=int,   default=128)
   +    parser.add_argument('-pu', '--pro_update_inters',    type=int,   default=3)
   +    parser.add_argument('-lu', '--lig_update_iters',     type=int,   default=10)
   +    parser.add_argument('-pl', '--pro_lig_update_iters', type=int,   default=1)
   +
   +    # Eğitim
   +    parser.add_argument('--lr',               type=float, default=1e-4)
   +    parser.add_argument('--clip_norm',        type=float, default=1.0)
   +    parser.add_argument('--epoch',            type=int,   default=1000)
   +    parser.add_argument('--batch_size',       type=int,   default=24)
   +    parser.add_argument('--anneal_iter',      type=int,   default=50000)
   +    parser.add_argument('--save_iter',        type=int,   default=10000)
   +    parser.add_argument('--print_iter',       type=int,   default=1000)
   +    parser.add_argument('--load_checkpoint',  default=None)
   +    parser.add_argument('--initial_step',     type=int,   default=0)
   +    parser.add_argument('--binder_per_batch', type=int,   default=2)
   +    parser.add_argument('--decoy_count',      type=int,   default=11)
   @@ -65,6 +65,5 @@
   -    os.makedirs(args.save_dir, exist_ok=True) 
   -
   -    # 1. WANDB BAŞLATMA
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    os.makedirs(args.save_dir, exist_ok=True)
   +
   +    wandb.init(project="PLANET_TRAINING",
   +               name=f"esm2_vector_gating_pkl_bs{args.batch_size}", config=args)
   +
   @@ -74,16 +73,31 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   -    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True)
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   ... (+375 satir daha)

--- PLANET_train_esm2_vector_gating_contrastive_loss.py : 306 satir | +231 -218 (base'e gore)
   @@ -9,0 +10 @@
   +from scipy.stats import pearsonr, spearmanr
   @@ -18,5 +19,3 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   -    
   -    # RDKit uyarılarını kapat
   -    RDLogger.DisableLog('rdApp.*') 
   +    from PLANET_datautils_esm2_vector import ProLigDataset          # ← LMDB versiyonu
   +    from PLANET_model_esm2_vector_gating_contrastive_loss import PLANET
   +    RDLogger.DisableLog('rdApp.*')
   @@ -27 +25,0 @@
   -# --- YARDIMCI FONKSİYONLAR ---
   @@ -29 +26,0 @@
   -    """Veri yapısı içindeki tüm Tensor'ları hedef cihaza taşır."""
   @@ -38,14 +35,35 @@
   -    # Dosya Yolları
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   -    parser.add_argument('-d','--save_dir', required=True, help="Model kayıt dizini")
   -
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   -    parser.add_argument('-n','--nheads', type=int, default=8)
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   -    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
   -    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
   -    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)
   +
   +    parser.add_argument('-t',  '--train',             required=True)
   +    parser.add_argument('-v',  '--valid',             required=True)
   +    parser.add_argument('-te', '--test',              required=True)
   +    parser.add_argument('-d',  '--save_dir',          required=True)
   +
   +    parser.add_argument('-s',  '--feature_dims',         type=int,   default=512)
   +    parser.add_argument('-n',  '--nheads',               type=int,   default=8)
   +    parser.add_argument('-k',  '--key_dims',             type=int,   default=128)
   +    parser.add_argument('-va', '--value_dims',           type=int,   default=128)
   +    parser.add_argument('-pu', '--pro_update_inters',    type=int,   default=3)
   +    parser.add_argument('-lu', '--lig_update_iters',     type=int,   default=10)
   +    parser.add_argument('-pl', '--pro_lig_update_iters', type=int,   default=1)
   +
   +    parser.add_argument('--lr',               type=float, default=1e-4)
   +    parser.add_argument('--clip_norm',        type=float, default=1.0)
   +    parser.add_argument('--epoch',            type=int,   default=250)
   +    parser.add_argument('--batch_size',       type=int,   default=32)
   +    parser.add_argument('--binder_per_batch', type=int,   default=16)
   +    parser.add_argument('--decoy_count',      type=int,   default=10)
   +    parser.add_argument('--anneal_iter',      type=int,   default=20000)
   +    parser.add_argument('--save_iter',        type=int,   default=10000)
   +    parser.add_argument('--print_iter',       type=int,   default=1000)
   +    parser.add_argument('--load_checkpoint',  default=None)
   +    parser.add_argument('--initial_step',     type=int,   default=0)
   +    parser.add_argument('--num_workers',      type=int,   default=12)
   +
   +    args = parser.parse_args()
   +    os.makedirs(args.save_dir, exist_ok=True)
   +
   +    wandb.init(
   +        project="PLANET_TRAINING",
   +        name=f"esm2_vg_contrastive_b{args.binder_per_batch}_d{args.batch_size - args.binder_per_batch}",
   +        config=vars(args)
   +    )
   @@ -53,18 +70,0 @@
   -    # Eğitim Ayarları
   -    parser.add_argument('--lr', type=float, default=1e-4)
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   -    parser.add_argument('--epoch', type=int, default=250)
   -    parser.add_argument('--batch_size', type=int, default=16)
   -    parser.add_argument('--anneal_iter', type=int, default=20000)
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   -    parser.add_argument('--print_iter', type=int, default=1000)
   -    parser.add_argument('--load_checkpoint', default=None)
   -    parser.add_argument('--initial_step', type=int, default=0)
   -
   -    args = parser.parse_args()
   -    os.makedirs(args.save_dir, exist_ok=True) 
   -
   -    # 1. WANDB BAŞLATMA
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   @@ -74,16 +74,18 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   -    valid_dataset = ProLigDataset(args.valid, args.batch_size, shuffle=False, decoy_flag=True)
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   ... (+387 satir daha)

--- PLANET_train_esm2_vector_gating_smooth.py : 296 satir | +93 -90 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle # <--- EKLENDİ (Dosyayı okumak için şart)
   @@ -18,2 +19,2 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   +    from PLANET_datautils_esm2_vector_smooth import ProLigDataset 
   +    from PLANET_model_esm2_vector_gating import PLANET 
   @@ -39,3 +40,3 @@
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   +    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   +    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   +    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   @@ -44,2 +45,2 @@
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   +    # Model Hiperparametreleri (ESM-2 İÇİN 512 YAPILDI)
   +    parser.add_argument('-s','--feature_dims', type=int, default=512, help="ESM-2 boyutu")
   @@ -47,2 +48,2 @@
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   +    parser.add_argument('-k','--key_dims', type=int, default=128)
   +    parser.add_argument('-va','--value_dims', type=int, default=128)
   @@ -55 +56 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0) 
   @@ -57 +58 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   +    parser.add_argument('--batch_size', type=int, default=32)
   @@ -59 +60 @@
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--save_iter', type=int, default=5-10000) 
   @@ -68,3 +69,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_ESM2", name=f"ESM_512_BS{args.batch_size}", config=args)
   +
   +    # 2. CİHAZ
   @@ -74 +75,9 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   +    # 3. VERİ YÜKLEME (BURASI DÜZELTİLDİ)
   +    # ProLigDataset string path değil, liste istiyor. O yüzden önce biz yüklüyoruz.
   +    print("📂 Veri listeleri yükleniyor...")
   +    with open(args.train, 'rb') as f: train_list = pickle.load(f)
   +    with open(args.valid, 'rb') as f: valid_list = pickle.load(f)
   +    with open(args.test, 'rb') as f:  test_list = pickle.load(f)
   +
   +    # --- 🌟 DÜZELTME BURADA 🌟 ---
   +    # valid_list yerine args.valid, test_list yerine args.test gönderiyoruz.
   @@ -76,8 +85,25 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   +    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   +    
   +    train_dataset = ProLigDataset(
   +        args.train, 
   +        args.batch_size, 
   +        shuffle=True, 
   +        decoy_flag=True, 
   +        binder_per_batch=2 
   +    )
   +
   +    # DataLoaders
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    
   +    train_loader = DataLoader(
   +        train_dataset,
   +        batch_size=1, # ProLigDataset kendi içinde batch'liyor
   +        num_workers=20,
   +        pin_memory=True,
   +        prefetch_factor=4,
   +        shuffle=True, # Veri düzeyinde rastgelelik
   +        collate_fn=lambda x: x[0]
   +    )
   +    
   +    print(f"   Train: {len(train_list)} | Valid: {len(valid_list)} | Test: {len(test_list)}")
   @@ -89 +115 @@
   -    # Parametre Başlatma (Initialization)
   +    # Parametre Başlatma
   @@ -124 +150 @@
   -                print(f"⚠️ Veri Yükleme Hatası (Step {total_step}): {e}", file=sys.stderr)
   ... (+124 satir daha)

--- PLANET_train_esm2_vector_gating_smooth_contrastive_loss.py : 296 satir | +93 -90 (base'e gore)
   @@ -6,0 +7 @@
   +import pickle # <--- EKLENDİ (Dosyayı okumak için şart)
   @@ -18,2 +19,2 @@
   -    from PLANET_datautils import ProLigDataset 
   -    from PLANET_model import PLANET 
   +    from PLANET_datautils_esm2_vector_smooth import ProLigDataset 
   +    from PLANET_model_esm2_vector_gating_contrastive_loss import PLANET 
   @@ -39,3 +40,3 @@
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   +    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   +    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   +    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   @@ -44,2 +45,2 @@
   -    # Model Hiperparametreleri
   -    parser.add_argument('-s','--feature_dims', type=int, default=300)
   +    # Model Hiperparametreleri (ESM-2 İÇİN 512 YAPILDI)
   +    parser.add_argument('-s','--feature_dims', type=int, default=512, help="ESM-2 boyutu")
   @@ -47,2 +48,2 @@
   -    parser.add_argument('-k','--key_dims', type=int, default=300)
   -    parser.add_argument('-va','--value_dims', type=int, default=300)
   +    parser.add_argument('-k','--key_dims', type=int, default=128)
   +    parser.add_argument('-va','--value_dims', type=int, default=128)
   @@ -55 +56 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0) 
   @@ -57 +58 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   +    parser.add_argument('--batch_size', type=int, default=32)
   @@ -59 +60 @@
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--save_iter', type=int, default=5-10000) 
   @@ -68,3 +69,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_ESM2", name=f"ESM_512_BS{args.batch_size}", config=args)
   +
   +    # 2. CİHAZ
   @@ -74 +75,9 @@
   -    # Validasyon & Test (Shuffle False, Decoy True)
   +    # 3. VERİ YÜKLEME (BURASI DÜZELTİLDİ)
   +    # ProLigDataset string path değil, liste istiyor. O yüzden önce biz yüklüyoruz.
   +    print("📂 Veri listeleri yükleniyor...")
   +    with open(args.train, 'rb') as f: train_list = pickle.load(f)
   +    with open(args.valid, 'rb') as f: valid_list = pickle.load(f)
   +    with open(args.test, 'rb') as f:  test_list = pickle.load(f)
   +
   +    # --- 🌟 DÜZELTME BURADA 🌟 ---
   +    # valid_list yerine args.valid, test_list yerine args.test gönderiyoruz.
   @@ -76,8 +85,25 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0]) 
   -
   -    test_dataset = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   -    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
   -
   -    # Eğitim (Shuffle True, Decoy True, Binder Oranı Önemli)
   -    train_dataset = ProLigDataset(args.train, args.batch_size, shuffle=True, decoy_flag=True, binder_per_batch=2)
   -    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   +    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True)
   +    
   +    train_dataset = ProLigDataset(
   +        args.train, 
   +        args.batch_size, 
   +        shuffle=True, 
   +        decoy_flag=True, 
   +        binder_per_batch=2 
   +    )
   +
   +    # DataLoaders
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    
   +    train_loader = DataLoader(
   +        train_dataset,
   +        batch_size=1, # ProLigDataset kendi içinde batch'liyor
   +        num_workers=20,
   +        pin_memory=True,
   +        prefetch_factor=4,
   +        shuffle=True, # Veri düzeyinde rastgelelik
   +        collate_fn=lambda x: x[0]
   +    )
   +    
   +    print(f"   Train: {len(train_list)} | Valid: {len(valid_list)} | Test: {len(test_list)}")
   @@ -89 +115 @@
   -    # Parametre Başlatma (Initialization)
   +    # Parametre Başlatma
   @@ -124 +150 @@
   -                print(f"⚠️ Veri Yükleme Hatası (Step {total_step}): {e}", file=sys.stderr)
   ... (+124 satir daha)

--- PLANET_train_noESM_noUW.py : 312 satir | +215 -196 (base'e gore)
   @@ -4,2 +4,4 @@
   -from torch.utils.data import DataLoader
   -import argparse, os, sys
   +import torch.optim.lr_scheduler as lr_scheduler
   +from torch.utils.data import DataLoader, Dataset
   +from torch.utils.data.dataloader import default_collate
   +import argparse, os, sys, pickle
   @@ -6,0 +9 @@
   +import pandas as pd
   @@ -10,2 +13,3 @@
   -
   -# --- İMPORT AYARLARI ---
   +from scipy.stats import pearsonr, spearmanr
   +
   +# --- PYTHON PATH GÜNCELLEMESİ ---
   @@ -13,0 +18 @@
   +
   @@ -16 +21,3 @@
   -
   +# --------------------------------
   +
   +# --- IMPORTLAR VE HATA YÖNETİMİ ---
   @@ -20,2 +27,2 @@
   -    
   -    # RDKit uyarılarını kapat
   +    from chemutils import ComplexPocket, tensorize_all, role_of_5
   +    
   @@ -22,0 +30 @@
   +    
   @@ -26,2 +34,3 @@
   -
   -# --- YARDIMCI FONKSİYONLAR ---
   +# ----------------------------------
   +
   +# PyTorch'a özel Tensor taşıma yardımcı fonksiyonu
   @@ -29 +37,0 @@
   -    """Veri yapısı içindeki tüm Tensor'ları hedef cihaza taşır."""
   @@ -36,0 +45 @@
   +    
   @@ -38,7 +47,7 @@
   -    # Dosya Yolları
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu")
   -    parser.add_argument('-te','--test', required=True, help="Core.pkl (Test) yolu")
   -    parser.add_argument('-d','--save_dir', required=True, help="Model kayıt dizini")
   -
   -    # Model Hiperparametreleri
   +    # Argümanlar
   +    parser.add_argument('-t','--train', required=True, help="Yol Train.pkl dosyasına.")
   +    parser.add_argument('-v','--valid',required=True, help="Yol Valid.pkl dosyasına.")
   +    parser.add_argument('-te','--test',required=True, help="Yol Core.pkl dosyasına.")
   +    parser.add_argument('-d','--save_dir', required=True, help="Model ağırlıklarının kaydedileceği dizin.")
   +
   +    # Model Parametreleri
   @@ -48,0 +58 @@
   +
   @@ -50,4 +60,3 @@
   -    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
   -    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)
   -    
   -    # Eğitim Ayarları
   +    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
   +    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
   +    
   @@ -55 +64,2 @@
   -    parser.add_argument('--clip_norm', type=float, default=10.0) # Güvenli değer
   +    parser.add_argument('--clip_norm', type=float, default=1.0)
   +
   @@ -57,3 +67,5 @@
   -    parser.add_argument('--batch_size', type=int, default=16)
   -    parser.add_argument('--anneal_iter', type=int, default=20000)
   -    parser.add_argument('--save_iter', type=int, default=10000) # Her 1000 adımda bir kaydet
   +    parser.add_argument('--batch_size',type=int,default=16)
   +    parser.add_argument('--anneal_iter', type=int, default=50000)
   +    
   +    parser.add_argument('--initial_step',type=int,default=0)
   +    parser.add_argument('--save_iter', type=int, default=10000)
   @@ -62 +74 @@
   -    parser.add_argument('--initial_step', type=int, default=0)
   +    parser.add_argument('--binder_per_batch', type=int, default=2)
   @@ -65,0 +78 @@
   +    print(args)
   @@ -68,3 +81,3 @@
   -    wandb.init(project="PLANET_Training", name=f"Final_Fix_BS{args.batch_size}", config=args)
   -
   -    # 2. CİHAZ VE VERİ YÜKLEYİCİLER
   +    wandb.init(project="PLANET_TRAINING", name=f"Noesm2_NoUW{args.batch_size}", config=args)
   +
   +    # 2. Cihaz Ayarı
   @@ -72,3 +85,3 @@
   ... (+367 satir daha)

==========================================================================================
GROUP: Scripts/ MODEL contrastive variants (base=PLANET_model_esm2_vector_gating_contrastive_loss.py)
==========================================================================================
BASE Scripts/PLANET_model_esm2_vector_gating_contrastive_loss.py: 208 satir

--- PLANET_model_esm2_vector_gating_contrastive_loss_mean.py : 186 satir | +55 -77 (base'e gore)
   @@ -2,0 +3 @@
   +import torch.nn.functional as F
   @@ -4 +4,0 @@
   -# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
   @@ -15 +14,0 @@
   -        # 1. Katmanlar (Vector + Gating + Dropout)
   @@ -18,2 +16,0 @@
   -        
   -        # ProLig: Vector Attention için koordinatları işleyen versiyon
   @@ -22,7 +19,4 @@
   -        # 2. Uncertainty Weighting Parametreleri
   -        # Loss'ları dengelemek için öğrenilebilir sigma parametreleri
   -        self.log_sigma_lig = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_pl = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_affinity = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        # Contrastive loss, affinity ile aynı sigma'yı paylaşıyor
   -        # Ayrı bir log_sigma_contrastive YOK
   +        self.log_sigma_lig        = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_pl         = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_affinity   = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   @@ -30,2 +24 @@
   -        # 3. Kayıp Fonksiyonları
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
   +        self.ligand_interaction_loss  = nn.BCELoss(reduction='sum')
   @@ -33 +26 @@
   -        self.mse_loss = nn.MSELoss(reduction='none')
   +        self.mse_loss                 = nn.MSELoss(reduction='none')
   @@ -36,6 +28,0 @@
   -        """
   -        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
   -        res_batch: Protein verisi
   -        mol_batch: Ligand verisi (Koordinatlar dahil)
   -        """
   -        # --- A. Protein Verisini Hazırla ---
   @@ -43 +30 @@
   -        fresidues = fresidues.to(self.device)
   +        fresidues         = fresidues.to(self.device)
   @@ -46,2 +32,0 @@
   -        # --- B. Ligand Verisini Hazırla ---
   -        # 6 elemanlı tuple kontrolü (Koordinat var mı?)
   @@ -51 +36 @@
   -            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi.")
   +            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor, ancak {len(mol_batch)} eleman geldi.")
   @@ -53,4 +38,4 @@
   -        fatoms = fatoms.to(self.device)
   -        fbonds = fbonds.to(self.device)
   -        agraph = agraph.to(self.device)
   -        bgraph = bgraph.to(self.device)
   +        fatoms          = fatoms.to(self.device)
   +        fbonds          = fbonds.to(self.device)
   +        agraph          = agraph.to(self.device)
   +        bgraph          = bgraph.to(self.device)
   @@ -59,3 +43,0 @@
   -        # --- C. Alt Katmanlardan Geçiş ---
   -        
   -        # 1. Protein Embedding (Gating Var)
   @@ -63,5 +45,2 @@
   -        
   -        # 2. Ligand Embedding
   -        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   -         
   -        # 3. Protein-Ligand Etkileşimi (Vector Attention Var)
   +        fatoms    = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   +
   @@ -79 +57,0 @@
   -        # targets train'de 6 elemanlı (group_ids var), eval'da 5 elemanlı
   @@ -84 +61,0 @@
   -            # Eval'da tüm moleküller tek group sayılır
   @@ -85,0 +63 @@
   +
   @@ -93 +71 @@
   -        # 1. Etkileşim Kayıpları
   +        # ── 1. Etkileşim Kayıpları ────────────────────────────────────────────
   @@ -95 +73 @@
   -        lig_interaction_loss, pro_lig_interaction_loss = [], []
   +        lig_interaction_loss, pro_lig_interaction_loss   = [], []
   @@ -98 +76 @@
   -            lig_total_points = atom_count**2
   +            lig_total_points = atom_count ** 2
   @@ -114,2 +92,2 @@
   -            lig_interaction_count      += lig_total_points
   -            pro_lig_interaction_count  += pl_total_points
   +            lig_interaction_count     += lig_total_points
   +            pro_lig_interaction_count += pl_total_points
   @@ -120,2 +98,4 @@
   -        # 2. Affinity Loss (MSE)
   -        affinity_mse = torch.sum(self.mse_loss(predicted_affinities, pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)
   +        # ── 2. Affinity Loss ──────────────────────────────────────────────────
   +        affinity_mse = torch.sum(
   ... (+82 satir daha)

--- PLANET_model_esm2_vector_gating_contrastive_loss_v2.py : 219 satir | +74 -63 (base'e gore)
   @@ -2,0 +3 @@
   +import torch.nn.functional as F
   @@ -23,6 +24,5 @@
   -        # Loss'ları dengelemek için öğrenilebilir sigma parametreleri
   -        self.log_sigma_lig = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_pl = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_affinity = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        # Contrastive loss, affinity ile aynı sigma'yı paylaşıyor
   -        # Ayrı bir log_sigma_contrastive YOK
   +        self.log_sigma_lig        = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_pl         = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_affinity   = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        # v2: Contrastive loss için ayrı sigma — pro-lig interaction skalasında bağımsız dengeleme
   +        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   @@ -31 +31 @@
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
   +        self.ligand_interaction_loss  = nn.BCELoss(reduction='sum')
   @@ -33 +33 @@
   -        self.mse_loss = nn.MSELoss(reduction='none')
   +        self.mse_loss                 = nn.MSELoss(reduction='none')
   @@ -43 +43 @@
   -        fresidues = fresidues.to(self.device)
   +        fresidues         = fresidues.to(self.device)
   @@ -47 +46,0 @@
   -        # 6 elemanlı tuple kontrolü (Koordinat var mı?)
   @@ -53,4 +52,4 @@
   -        fatoms = fatoms.to(self.device)
   -        fbonds = fbonds.to(self.device)
   -        agraph = agraph.to(self.device)
   -        bgraph = bgraph.to(self.device)
   +        fatoms          = fatoms.to(self.device)
   +        fbonds          = fbonds.to(self.device)
   +        agraph          = agraph.to(self.device)
   +        bgraph          = bgraph.to(self.device)
   @@ -60,2 +58,0 @@
   -        
   -        # 1. Protein Embedding (Gating Var)
   @@ -63,5 +60,2 @@
   -        
   -        # 2. Ligand Embedding
   -        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   -         
   -        # 3. Protein-Ligand Etkileşimi (Vector Attention Var)
   +        fatoms    = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   +
   @@ -84 +77,0 @@
   -            # Eval'da tüm moleküller tek group sayılır
   @@ -85,0 +79 @@
   +
   @@ -93 +87 @@
   -        # 1. Etkileşim Kayıpları
   +        # ── 1. Etkileşim Kayıpları (değişmedi) ───────────────────────────────
   @@ -95 +89 @@
   -        lig_interaction_loss, pro_lig_interaction_loss = [], []
   +        lig_interaction_loss, pro_lig_interaction_loss   = [], []
   @@ -98 +92 @@
   -            lig_total_points = atom_count**2
   +            lig_total_points = atom_count ** 2
   @@ -114,2 +108,2 @@
   -            lig_interaction_count      += lig_total_points
   -            pro_lig_interaction_count  += pl_total_points
   +            lig_interaction_count     += lig_total_points
   +            pro_lig_interaction_count += pl_total_points
   @@ -120,5 +114,30 @@
   -        # 2. Affinity Loss (MSE)
   -        affinity_mse = torch.sum(self.mse_loss(predicted_affinities, pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)
   -
   -        # 3. Contrastive Loss — group_id'ye göre protein-spesifik
   -        # Her pocket kendi içinde: binder skoru > decoy skoru olmalı
   +        # ── 2. Affinity Loss (değişmedi) ─────────────────────────────────────
   +        affinity_mse = torch.sum(
   +            self.mse_loss(predicted_affinities, pKs) * pK_flags
   +        ) / (torch.sum(pK_flags) + 1e-6)
   +
   +        # ── 3. Contact-tabanlı Contrastive Loss (v3) ─────────────────────────
   +        # Her molekül için predicted_interactions (atom×residue matrisi) üzerinde
   +        # softmax-weighted mean alınır — affinity hesabındaki ağırlıklı toplam
   +        # mantığıyla tutarlı.
   +        #
   +        # Neden softmax-weighted mean?
   +        #   - Düz mean: yüksek contact skorları düşük skorların arasında boğulur
   +        #     (1000 çiftin ortalaması ≈ 0.05, signal kaybolur)
   +        #   - Softmax-weighted: yüksek skorlu çiftler daha fazla ağırlık alır,
   +        #     binder/decoy ayrımı netleşir — affinity'deki interaction_weight ile aynı fikir
   +        #
   +        # margin=0.1: softmax-weighted değerler 0-1 arasında ama daha yüksek
   +        #   seviyelerde olacak, 0.1'lik fark makul bir hedef.
   +
   +        contact_per_mol = []
   +        ptr = 0
   ... (+77 satir daha)

--- PLANET_model_esm2_vector_gating_contrastive_loss_v3.py : 221 satir | +18 -5 (base'e gore)
   @@ -124 +124,4 @@
   -        # Her pocket kendi içinde: binder skoru > decoy skoru olmalı
   +        # v3: margin = 3.0 + hard negative mining
   +        # Affinity skor aralığı genelde 4-12 (pK), 1.0'lık margin çok küçük kalıyor.
   +        # Ek olarak: tüm decoy'lar yerine sadece "zor" olanlara loss uygula —
   +        # binder skoruna yakın ya da daha yüksek tahmin alan decoy'lar.
   @@ -126,0 +130 @@
   +        margin = 3.0  # v1: 1.0 → v3: 3.0 (affinity skalasına uygun)
   @@ -137,4 +141,13 @@
   -                diff = g_decoy_preds.unsqueeze(1) - g_active_preds.unsqueeze(0)
   -                margin = 1.0
   -                # margin offset ile:
   -                group_losses.append(torch.mean(torch.nn.functional.softplus(diff - margin)))
   +                # Tüm (decoy, active) çiftleri
   +                diff = g_decoy_preds.unsqueeze(1) - g_active_preds.unsqueeze(0)  # (n_decoy, n_active)
   +
   +                # Hard negative mining:
   +                # diff > -margin  →  decoy_pred > active_pred - margin
   +                # Yani margin'in tatmin edilmediği zor çiftler.
   +                # diff < -margin  →  zaten yeterince ayrılmış, gradient harcama.
   +                hard_mask = diff > -margin
   +
   +                if hard_mask.any():
   +                    hard_diffs = diff[hard_mask]
   +                    group_losses.append(torch.mean(torch.nn.functional.softplus(hard_diffs - margin)))
   +                # Eğer hiç hard pair yoksa bu group için loss = 0 (zaten iyi ayrılmış)

--- PLANET_model_esm2_vector_gating_contrastive_loss_v4.py : 185 satir | +48 -71 (base'e gore)
   @@ -2,0 +3 @@
   +import torch.nn.functional as F
   @@ -4 +4,0 @@
   -# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
   @@ -6 +6 @@
   -from layers_esm2_vector_gating import ProteinEGNN, LigandGAT, ProLig 
   +from layers_esm2_vector_gating_projection_head import ProteinEGNN, LigandGAT, ProLig
   @@ -14,2 +14 @@
   -        
   -        # 1. Katmanlar (Vector + Gating + Dropout)
   +
   @@ -18,2 +16,0 @@
   -        
   -        # ProLig: Vector Attention için koordinatları işleyen versiyon
   @@ -22,7 +19,6 @@
   -        # 2. Uncertainty Weighting Parametreleri
   -        # Loss'ları dengelemek için öğrenilebilir sigma parametreleri
   -        self.log_sigma_lig = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_pl = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        self.log_sigma_affinity = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   -        # Contrastive loss, affinity ile aynı sigma'yı paylaşıyor
   -        # Ayrı bir log_sigma_contrastive YOK
   +        # Uncertainty Weighting parametreleri
   +        self.log_sigma_lig         = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_pl          = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        self.log_sigma_affinity    = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   +        # v4: Contrastive için ayrı sigma — latent uzayda bağımsız sinyal
   +        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
   @@ -30,2 +26 @@
   -        # 3. Kayıp Fonksiyonları
   -        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
   +        self.ligand_interaction_loss  = nn.BCELoss(reduction='sum')
   @@ -33 +28 @@
   -        self.mse_loss = nn.MSELoss(reduction='none')
   +        self.mse_loss                 = nn.MSELoss(reduction='none')
   @@ -36,6 +30,0 @@
   -        """
   -        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
   -        res_batch: Protein verisi
   -        mol_batch: Ligand verisi (Koordinatlar dahil)
   -        """
   -        # --- A. Protein Verisini Hazırla ---
   @@ -46,2 +34,0 @@
   -        # --- B. Ligand Verisini Hazırla ---
   -        # 6 elemanlı tuple kontrolü (Koordinat var mı?)
   @@ -51 +38 @@
   -            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi.")
   +            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor, ancak {len(mol_batch)} eleman geldi.")
   @@ -59,3 +45,0 @@
   -        # --- C. Alt Katmanlardan Geçiş ---
   -        
   -        # 1. Protein Embedding (Gating Var)
   @@ -63,6 +47,4 @@
   -        
   -        # 2. Ligand Embedding
   -        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   -         
   -        # 3. Protein-Ligand Etkileşimi (Vector Attention Var)
   -        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
   +        fatoms    = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
   +
   +        # v4: 4 çıktı (ek olarak z_complexes)
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities, z_complexes = self.prolig(
   @@ -71,3 +53,3 @@
   -        
   -        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
   -    
   +
   +        return (predicted_lig_interactions, predicted_interactions, predicted_affinities, z_complexes)
   +
   @@ -75 +57,2 @@
   -        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions
   +        # v4: 4 çıktı unpack
   +        predicted_lig_interactions, predicted_interactions, predicted_affinities, z_complexes = predictions
   @@ -79 +61,0 @@
   -        # targets train'de 6 elemanlı (group_ids var), eval'da 5 elemanlı
   @@ -84 +65,0 @@
   -            # Eval'da tüm moleküller tek group sayılır
   @@ -85,0 +67 @@
   +
   @@ -114,2 +96,2 @@
   -            lig_interaction_count      += lig_total_points
   -            pro_lig_interaction_count  += pl_total_points
   +            lig_interaction_count     += lig_total_points
   +            pro_lig_interaction_count += pl_total_points
   @@ -123,2 +105,3 @@
   -        # 3. Contrastive Loss — group_id'ye göre protein-spesifik
   -        # Her pocket kendi içinde: binder skoru > decoy skoru olmalı
   +        # 3. Latent-space Contrastive Loss (v4)
   +        # z_complexes: (N_mol, 128) — her molekül için L2-normalized projection embedding
   ... (+64 satir daha)

==========================================================================================
GROUP: Scripts/ TRAIN contrastive variants (base=PLANET_train_esm2_vector_gating_contrastive_loss.py)
==========================================================================================
BASE Scripts/PLANET_train_esm2_vector_gating_contrastive_loss.py: 330 satir

--- PLANET_train_esm2_vector_gating_contrastive_loss_mean.py : 328 satir | +77 -79 (base'e gore)
   @@ -14,3 +14,2 @@
   -datautils_path = current_dir
   -if datautils_path not in sys.path:
   -    sys.path.append(datautils_path)
   +if current_dir not in sys.path:
   +    sys.path.append(current_dir)
   @@ -19,2 +18,3 @@
   -    from PLANET_datautils_esm2_vector import ProLigDataset          # ← LMDB versiyonu
   -    from PLANET_model_esm2_vector_gating_contrastive_loss_v3 import PLANET
   +    from PLANET_datautils_esm2_vector import ProLigDataset
   +    # v2: contact-tabanlı contrastive loss model dosyası
   +    from PLANET_model_esm2_vector_gating_contrastive_loss_mean import PLANET
   @@ -37,5 +37,5 @@
   -    # Dosya Yolları (LMDB klasör yolları)
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   -    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   -    parser.add_argument('-d',  '--save_dir', required=True, help="Model kayıt dizini")
   +    # Dosya Yolları
   +    parser.add_argument('-t',  '--train',    required=True)
   +    parser.add_argument('-v',  '--valid',    required=True)
   +    parser.add_argument('-te', '--test',     required=True)
   +    parser.add_argument('-d',  '--save_dir', required=True)
   @@ -57,4 +57,2 @@
   -    parser.add_argument('--binder_per_batch', type=int,   default=1,
   -                        help="Her batch'te kaç pocket: toplam molekül = binder_per_batch * 12")
   -    parser.add_argument('--decoy_count',      type=int,   default=10,
   -                        help="Valid/test'te her binder için max kaç decoy alınsın")
   +    parser.add_argument('--binder_per_batch', type=int,   default=1)
   +    parser.add_argument('--decoy_count',      type=int,   default=10)
   @@ -72 +70 @@
   -        name=f"esm_uw_vector_gating_contrastive_margin3_finetune"
   +        name=f"esm_uw_vector_gating_contrastive_v2_mean_m0"
   @@ -75 +72,0 @@
   -    # 2. CİHAZ
   @@ -80,2 +77 @@
   -    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
   -    
   +    test_dataset  = ProLigDataset(args.test,  args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
   @@ -83,4 +79,2 @@
   -        args.train,
   -        args.batch_size,
   -        shuffle=True,
   -        decoy_flag=True,
   +        args.train, args.batch_size,
   +        shuffle=True, decoy_flag=True,
   @@ -90,4 +84,2 @@
   -    # DataLoaders
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   -    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   -    
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
   +    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
   @@ -95,8 +87,4 @@
   -        train_dataset,
   -        batch_size=1,
   -        num_workers=6,        # 5 → 12 (cpus-per-task=20, 12 worker + 4 eval + 4 ana process)
   -        pin_memory=True,       # False → True (CPU→GPU transfer hızlanır)
   -        prefetch_factor=8,     # 4 → 8 (daha fazla batch önceden hazırlanır)
   -        persistent_workers=True,  # False → True (worker'lar epoch sonunda ölmesin)
   -        shuffle=True,
   -        collate_fn=lambda x: x[0]
   +        train_dataset, batch_size=1,
   +        num_workers=4, pin_memory=True,
   +        prefetch_factor=4, persistent_workers=True,
   +        shuffle=True, collate_fn=lambda x: x[0]
   @@ -107 +95 @@
   -    # 5. MODEL
   +    # MODEL
   @@ -121 +108,0 @@
   -    # 6. OPTİMİZER
   @@ -128,4 +115,12 @@
   -        PLANET_model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
   -
   -    total_step = args.initial_step
   -    meters     = np.zeros(6)
   +        # v2 modeline yeni log_sigma_contrastive parametresi eklendi.
   +        # Eski checkpoint'ten yüklerken strict=False — eksik anahtar sorun değil,
   +        # yeni parametre xavier init ile başlar.
   +        state = torch.load(args.load_checkpoint, map_location=device)
   +        missing, unexpected = PLANET_model.load_state_dict(state, strict=False)
   +        if missing:
   +            print(f"   ℹ️  Yeni parametreler (init edildi): {missing}")
   +        if unexpected:
   +            print(f"   ⚠️  Checkpoint'te fazladan anahtar (atlandı): {unexpected}")
   +
   +    total_step     = args.initial_step
   +    meters         = np.zeros(6)
   @@ -133 +128 @@
   -    beta       = 0.0
   ... (+98 satir daha)

--- PLANET_train_esm2_vector_gating_contrastive_loss_v2.py : 328 satir | +77 -79 (base'e gore)
   @@ -14,3 +14,2 @@
   -datautils_path = current_dir
   -if datautils_path not in sys.path:
   -    sys.path.append(datautils_path)
   +if current_dir not in sys.path:
   +    sys.path.append(current_dir)
   @@ -19,2 +18,3 @@
   -    from PLANET_datautils_esm2_vector import ProLigDataset          # ← LMDB versiyonu
   -    from PLANET_model_esm2_vector_gating_contrastive_loss_v3 import PLANET
   +    from PLANET_datautils_esm2_vector import ProLigDataset
   +    # v2: contact-tabanlı contrastive loss model dosyası
   +    from PLANET_model_esm2_vector_gating_contrastive_loss_v2 import PLANET
   @@ -37,5 +37,5 @@
   -    # Dosya Yolları (LMDB klasör yolları)
   -    parser.add_argument('-t','--train', required=True, help="Train.pkl yolu (train_esm2.pkl)")
   -    parser.add_argument('-v','--valid', required=True, help="Valid.pkl yolu (valid_esm2.pkl)")
   -    parser.add_argument('-te','--test', required=True, help="Test.pkl yolu (test_esm2.pkl)")
   -    parser.add_argument('-d',  '--save_dir', required=True, help="Model kayıt dizini")
   +    # Dosya Yolları
   +    parser.add_argument('-t',  '--train',    required=True)
   +    parser.add_argument('-v',  '--valid',    required=True)
   +    parser.add_argument('-te', '--test',     required=True)
   +    parser.add_argument('-d',  '--save_dir', required=True)
   @@ -57,4 +57,2 @@
   -    parser.add_argument('--binder_per_batch', type=int,   default=1,
   -                        help="Her batch'te kaç pocket: toplam molekül = binder_per_batch * 12")
   -    parser.add_argument('--decoy_count',      type=int,   default=10,
   -                        help="Valid/test'te her binder için max kaç decoy alınsın")
   +    parser.add_argument('--binder_per_batch', type=int,   default=1)
   +    parser.add_argument('--decoy_count',      type=int,   default=10)
   @@ -72 +70 @@
   -        name=f"esm_uw_vector_gating_contrastive_margin3_finetune"
   +        name=f"esm_uw_vector_gating_contrastive_v2_softmax_m0"
   @@ -75 +72,0 @@
   -    # 2. CİHAZ
   @@ -80,2 +77 @@
   -    test_dataset  = ProLigDataset(args.test, args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
   -    
   +    test_dataset  = ProLigDataset(args.test,  args.batch_size, shuffle=False, decoy_flag=True, decoy_count=args.decoy_count)
   @@ -83,4 +79,2 @@
   -        args.train,
   -        args.batch_size,
   -        shuffle=True,
   -        decoy_flag=True,
   +        args.train, args.batch_size,
   +        shuffle=True, decoy_flag=True,
   @@ -90,4 +84,2 @@
   -    # DataLoaders
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   -    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   -    
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
   +    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
   @@ -95,8 +87,4 @@
   -        train_dataset,
   -        batch_size=1,
   -        num_workers=6,        # 5 → 12 (cpus-per-task=20, 12 worker + 4 eval + 4 ana process)
   -        pin_memory=True,       # False → True (CPU→GPU transfer hızlanır)
   -        prefetch_factor=8,     # 4 → 8 (daha fazla batch önceden hazırlanır)
   -        persistent_workers=True,  # False → True (worker'lar epoch sonunda ölmesin)
   -        shuffle=True,
   -        collate_fn=lambda x: x[0]
   +        train_dataset, batch_size=1,
   +        num_workers=4, pin_memory=True,
   +        prefetch_factor=4, persistent_workers=True,
   +        shuffle=True, collate_fn=lambda x: x[0]
   @@ -107 +95 @@
   -    # 5. MODEL
   +    # MODEL
   @@ -121 +108,0 @@
   -    # 6. OPTİMİZER
   @@ -128,4 +115,12 @@
   -        PLANET_model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
   -
   -    total_step = args.initial_step
   -    meters     = np.zeros(6)
   +        # v2 modeline yeni log_sigma_contrastive parametresi eklendi.
   +        # Eski checkpoint'ten yüklerken strict=False — eksik anahtar sorun değil,
   +        # yeni parametre xavier init ile başlar.
   +        state = torch.load(args.load_checkpoint, map_location=device)
   +        missing, unexpected = PLANET_model.load_state_dict(state, strict=False)
   +        if missing:
   +            print(f"   ℹ️  Yeni parametreler (init edildi): {missing}")
   +        if unexpected:
   +            print(f"   ⚠️  Checkpoint'te fazladan anahtar (atlandı): {unexpected}")
   +
   +    total_step     = args.initial_step
   +    meters         = np.zeros(6)
   @@ -133 +128 @@
   -    beta       = 0.0
   ... (+98 satir daha)

--- PLANET_train_esm2_vector_gating_contrastive_loss_v4.py : 330 satir | +17 -17 (base'e gore)
   @@ -20 +20 @@
   -    from PLANET_model_esm2_vector_gating_contrastive_loss_v3 import PLANET
   +    from PLANET_model_esm2_vector_gating_contrastive_loss_v4 import PLANET
   @@ -72 +72 @@
   -        name=f"esm_uw_vector_gating_contrastive_margin3_finetune"
   +        name=f"esm_uw_vg_contrastive_v4_projection_scratch"
   @@ -91,2 +91,2 @@
   -    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0]) 
   -    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
   +    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0]) 
   +    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x:x[0])
   @@ -97 +97 @@
   -        num_workers=6,        # 5 → 12 (cpus-per-task=20, 12 worker + 4 eval + 4 ana process)
   +        num_workers=4,        # 5 → 12 (cpus-per-task=20, 12 worker + 4 eval + 4 ana process)
   @@ -99 +99 @@
   -        prefetch_factor=8,     # 4 → 8 (daha fazla batch önceden hazırlanır)
   +        prefetch_factor=4,     # 4 → 8 (daha fazla batch önceden hazırlanır)
   @@ -159,4 +159,5 @@
   -                # Uncertainty Weighting
   -                log_sigma_lig_c = torch.clamp(PLANET_model.log_sigma_lig,      min=-2, max=2)
   -                log_sigma_pl_c  = torch.clamp(PLANET_model.log_sigma_pl,       min=-2, max=2)
   -                log_sigma_aff_c = torch.clamp(PLANET_model.log_sigma_affinity, min=-2, max=2)
   +                # Uncertainty Weighting — v4: contrastive ayrı sigma
   +                log_sigma_lig_c = torch.clamp(PLANET_model.log_sigma_lig,         min=-2, max=2)
   +                log_sigma_pl_c  = torch.clamp(PLANET_model.log_sigma_pl,          min=-2, max=2)
   +                log_sigma_aff_c = torch.clamp(PLANET_model.log_sigma_affinity,    min=-2, max=2)
   +                log_sigma_con_c = torch.clamp(PLANET_model.log_sigma_contrastive, min=-2, max=2)
   @@ -167,3 +168,2 @@
   -                # Contrastive: affinity ile birlikte beta ile büyür
   -                # Beta=0 iken ikisi de kapalı, birlikte anlamlı hale gelir
   -                loss_con_w = torch.exp(-log_sigma_aff_c) * (beta * con_loss)
   +                # v4: contrastive kendi sigma'sıyla, bağımsız
   +                loss_con_w = torch.exp(-log_sigma_con_c) * (beta * con_loss) + log_sigma_con_c if beta > 0 else 0.0
   @@ -183 +182,0 @@
   -                print(f"⚠️ Eğitim Adımı Hatası: {e}")
   @@ -215,3 +214,4 @@
   -                        "Uncertainty/Sigma_Lig":      torch.exp(PLANET_model.log_sigma_lig).item(),
   -                        "Uncertainty/Sigma_PL":       torch.exp(PLANET_model.log_sigma_pl).item(),
   -                        "Uncertainty/Sigma_Affinity": torch.exp(PLANET_model.log_sigma_affinity).item(),
   +                        "Uncertainty/Sigma_Lig":         torch.exp(PLANET_model.log_sigma_lig).item(),
   +                        "Uncertainty/Sigma_PL":          torch.exp(PLANET_model.log_sigma_pl).item(),
   +                        "Uncertainty/Sigma_Affinity":    torch.exp(PLANET_model.log_sigma_affinity).item(),
   +                        "Uncertainty/Sigma_Contrastive": torch.exp(PLANET_model.log_sigma_contrastive).item(),

==========================================================================================
GROUP: Dataset_preparation/ DATAUTILS (base=PLANET_datautils.py)
==========================================================================================
BASE Dataset_preparation/PLANET_datautils.py: 180 satir

--- PLANET_datautils_esm2.py : 162 satir | +73 -91 (base'e gore)
   @@ -6,2 +6,2 @@
   -# Kendi modüllerin (Aynı dizinde olduğu varsayılıyor)
   -from chemutils import ComplexPocket, tensorize_all
   +# ESM2 uyumlu modülün
   +from chemutils_esm2 import tensorize_all 
   @@ -9,2 +9,3 @@
   -# JSON dosyasının yolu
   -PK_MAP_PATH_LOCAL = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
   +# --- KESİN YOLLAR ---
   +PK_MAP_JSON = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
   +ESM_SUBDIR = "planet_esm2_pkl"
   @@ -14 +15 @@
   -        # 1. ⚡ YEREL SSD AYARI: os.getlogin() yerine getpass.getuser() (HPC Uyumu)
   +        # 1. ⚡ YEREL SSD AYARI (HPC Uyumlu)
   @@ -17 +18 @@
   -        
   +
   @@ -19 +20 @@
   -            print(f"UYARI: {data_pickle_path} bulunamadı!")
   +            print(f"⚠️ {data_pickle_path} bulunamadı!")
   @@ -28 +29 @@
   -        # 2. ⚖️ DENGELEME MANTIĞI: Binder ve Decoy dengeli batch'ler oluşturur
   +        # 2. ⚖️ DENGELEME MANTIĞI
   @@ -33,21 +34,15 @@
   -            if len(binders) == 0:
   -                self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]
   -            else:
   -                decoys_per_batch = max(1, batch_size - binder_per_batch)
   -                num_batches = max(1, len(binders) // binder_per_batch)
   -                
   -                random.shuffle(binders)
   -                random.shuffle(decoys)
   -                
   -                from itertools import cycle
   -                decoy_cycle = cycle(decoys) if decoys else []
   -                
   -                balanced_batches = []
   -                for i in range(num_batches):
   -                    batch = binders[i*binder_per_batch : (i+1)*binder_per_batch]
   -                    if decoy_cycle:
   -                        batch += [next(decoy_cycle) for _ in range(decoys_per_batch)]
   -                    random.shuffle(batch)
   -                    balanced_batches.append(batch)
   -                self.data = balanced_batches
   -                print(f"✅ DATASET YÜKLENDİ: {len(self.data)} dengeli batch hazır.")
   +            num_batches = len(binders) // binder_per_batch if binders else 1
   +            random.shuffle(binders)
   +            
   +            from itertools import cycle
   +            decoy_cycle = cycle(decoys) if decoys else []
   +
   +            balanced_batches = []
   +            for i in range(num_batches):
   +                batch_binders = binders[i * binder_per_batch : (i + 1) * binder_per_batch]
   +                batch_decoys = [next(decoy_cycle) for _ in range(batch_size - binder_per_batch)] if decoys else []
   +                batch = batch_binders + batch_decoys
   +                random.shuffle(batch)
   +                balanced_batches.append(batch)
   +            self.data = balanced_batches
   +            print(f"✅ ESM2 DATASET HAZIR: {len(self.data)} Batch.")
   @@ -55 +50 @@
   -            self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]
   +            self.data = [full_data[i : i + batch_size] for i in range(0, len(full_data), batch_size)]
   @@ -57,10 +52 @@
   -    def __len__(self):
   -        return len(self.data)
   -
   -    def __getitem__(self, idx):
   -        try:
   -            return self.tensorize(idx)
   -        except Exception as e:
   -            # Hata durumunda batch'i atla ama logla
   -            # print(f"VERİ HATASI: {e}", file=sys.stderr)
   -            return None
   +    def __len__(self): return len(self.data)
   @@ -70 +55,0 @@
   -        
   @@ -72 +57 @@
   -            source_path = record[0] 
   +            source_path, pk_val = record[0], record[1]
   @@ -76,3 +61,3 @@
   -            # 3. 📂 BYTE OPTIMIZASYONU VE SSD TRANSFERİ
   -            if is_multi and "decoys.pkl" in file_name:
   -                byte_file_name = file_name.replace("_decoys.pkl", "_decoys_byte.pkl")
   +            # 3. 📂 ESM2 BYTE-OFFSET OPTIMIZASYONU
   +            if is_multi and "_esm512d.pkl" in source_path:
   +                byte_file_name = file_name.replace("_esm512d.pkl", "_byte_esm512d.pkl")
   @@ -86 +71 @@
   -            # --- 🛡️ AKILLI DISK KONTROLÜ (EN KRİTİK KISIM) ---
   +            # --- 🛡️ AKILLI DISK KONTROLÜ (10 GB EMNİYET KİLİDİ) ---
   ... (+107 satir daha)

--- PLANET_datautils_esm2_vector.py : 241 satir | +233 -172 (base'e gore)
   @@ -1,2 +1,18 @@
   -import os, pickle, random, shutil, sys, argparse, json, getpass
   -import pandas as pd
   +"""
   +PLANET_datautils_esm2_lmdb_v2.py
   +=================================
   +lmdb_planet_v2 ile çalışır.
   +
   +Her binder key'i → ComplexPocket(ligand=gerçek, decoys=[Mol1, Mol2, ...])
   +
   +Train modu:
   +- Batch başına sadece binder key'leri okunur
   +- tensorize_all() → random_ligand_decoy() → random decoy seçimi
   +- Hız: eski PKL sistemi ile aynı
   +
   +Eval modu (decoy_count verilince):
   +- Her binder için ilk min(decoy_count, mevcut) kadar decoy alınır
   +- Binder + decoylar deterministik olarak işlenir
   +"""
   +
   +import os, pickle, random, lmdb, json
   @@ -4 +20 @@
   -from tqdm import tqdm
   +import torch
   @@ -6,5 +22,2 @@
   -# Kendi modüllerin (Aynı dizinde olduğu varsayılıyor)
   -from chemutils import ComplexPocket, tensorize_all
   -
   -# JSON dosyasının yolu
   -PK_MAP_PATH_LOCAL = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
   +from chemutils_esm2_vector import tensorize_all, tensorize_protein_pocket, tensorize_ligand
   +
   @@ -13,22 +26,178 @@
   -    def __init__(self, data_pickle_path, batch_size, shuffle=True, decoy_flag=True, binder_per_batch=4):
   -        # 1. ⚡ YEREL SSD AYARI: os.getlogin() yerine getpass.getuser() (HPC Uyumu)
   -        self.local_ssd_dir = f"/tmp/{getpass.getuser()}_planet_cache"
   -        os.makedirs(self.local_ssd_dir, exist_ok=True)
   -        
   -        if not os.path.exists(data_pickle_path):
   -            print(f"UYARI: {data_pickle_path} bulunamadı!")
   -            self.data = []
   -            return
   -
   -        with open(data_pickle_path, 'rb') as f:
   -            full_data = pickle.load(f)
   -        
   -        self.decoy_flag = decoy_flag
   -
   -        # 2. ⚖️ DENGELEME MANTIĞI: Binder ve Decoy dengeli batch'ler oluşturur
   -        if shuffle and decoy_flag:
   -            binders = [r for r in full_data if float(r[1]) > 0.0]
   -            decoys = [r for r in full_data if float(r[1]) == 0.0]
   -            
   -            if len(binders) == 0:
   -                self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]
   +
   +    def __init__(self, lmdb_path, batch_size, shuffle=True,
   +                 decoy_flag=True, binder_per_batch=2, decoy_count=None):
   +
   +        self.lmdb_path        = lmdb_path
   +        self.batch_size       = batch_size
   +        self.decoy_flag       = decoy_flag
   +        self.binder_per_batch = binder_per_batch
   +        self.decoy_count      = decoy_count
   +
   +        # Fork-safe
   +        self._env = None
   +        self._txn = None
   +        self._pid = None
   +
   +        # ── Meta JSON ──────────────────────────────────────────
   +        meta_path = lmdb_path.rstrip('/\\')
   +        if meta_path.endswith('.lmdb'):
   +            meta_path = meta_path[:-5]
   +        meta_path += '_meta.json'
   +
   +        if not os.path.exists(meta_path):
   +            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')
   +
   +        with open(meta_path, 'r') as f:
   +            meta = json.load(f)
   +
   +        # v2'de sadece binder'lar var
   +        self.binders = meta
   +
   +        # ── Batch listesi ──────────────────────────────────────
   +        if decoy_count is not None:
   +            self.batches = self._build_eval_batches()
   +            mode = 'eval'
   +        else:
   ... (+321 satir daha)

--- PLANET_datautils_esm2_vector_new.py : 229 satir | +217 -168 (base'e gore)
   @@ -1,4 +1,2 @@
   -import os, pickle, random, shutil, sys, argparse, json, getpass
   -import pandas as pd
   -import numpy as np
   -from tqdm import tqdm
   +import os, pickle, random, lmdb, json
   +import torch
   @@ -6,5 +4,93 @@
   -# Kendi modüllerin (Aynı dizinde olduğu varsayılıyor)
   -from chemutils import ComplexPocket, tensorize_all
   -
   -# JSON dosyasının yolu
   -PK_MAP_PATH_LOCAL = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
   +
   +
   +def _merge_single_outputs(outputs_list):
   +    """
   +    Her biri tensorize_all([single_obj]) çıktısı olan listeyi
   +    tek bir batch çıktısına birleştirir.
   +
   +    outputs_list[i] = (res_batch_tuple, mol_batch_tuple, mol_int, pl_int, pK, pK_flag, label)
   +    """
   +    import torch
   +
   +    # ── res_feature_batch birleştir ──────────────────────────────────────────
   +    all_fresidues, all_res_scopes, all_alpha = [], [], []
   +    total_res = 0
   +    for out in outputs_list:
   +        fresidues, res_map, res_scope, alpha = out[0]
   +        n = fresidues.shape[0]
   +        all_fresidues.append(fresidues)
   +        all_alpha.append(alpha)
   +        all_res_scopes.append((total_res, n))
   +        total_res += n
   +
   +    # res_map: blok-diyagonal birleştirme
   +    combined_res_map = torch.zeros(total_res, total_res, 1)
   +    ptr = 0
   +    for out in outputs_list:
   +        _, res_map, _, _ = out[0]
   +        n = res_map.shape[0]
   +        combined_res_map[ptr:ptr+n, ptr:ptr+n, :] = res_map
   +        ptr += n
   +
   +    res_batch = (
   +        torch.cat(all_fresidues, dim=0),
   +        combined_res_map,
   +        all_res_scopes,
   +        torch.cat(all_alpha, dim=0),
   +    )
   +
   +    # ── mol_feature_batch birleştir ──────────────────────────────────────────
   +    all_fatoms, all_fbonds, all_agraphs, all_bgraphs, all_lig_scopes, all_lig_coords = [], [], [], [], [], []
   +    total_atoms = 0
   +    total_bonds = 1  # padding bond
   +
   +    for out in outputs_list:
   +        fatoms, fbonds, agraph, bgraph, lig_scope, lig_coords = out[1]
   +        n_atoms = fatoms.shape[0]
   +        n_bonds = fbonds.shape[0]   # padding dahil
   +
   +        all_fatoms.append(fatoms)
   +        all_lig_coords.append(lig_coords)
   +        all_lig_scopes.append((total_atoms, n_atoms))
   +
   +        # agraph: atom → bond indeksi, offset = total_bonds - 1 (padding çıkarılır)
   +        agraph_off = agraph.clone()
   +        mask_a = agraph_off != 0
   +        agraph_off[mask_a] += (total_bonds - 1)
   +        all_agraphs.append(agraph_off)
   +
   +        # bgraph: bond → bond indeksi, aynı offset
   +        bgraph_off = bgraph.clone()
   +        mask_b = bgraph_off != 0
   +        bgraph_off[mask_b] += (total_bonds - 1)
   +        all_bgraphs.append(bgraph_off[1:])
   +
   +        # fbonds: padding bond'u (index 0) atla, geri kalanı ekle
   +        all_fbonds.append(fbonds[1:])
   +
   +        total_atoms += n_atoms
   +        total_bonds += (n_bonds - 1)   # padding hariç gerçek bond sayısı
   +
   +    # Başa tek padding satırı ekle
   +    bond_dim = all_fbonds[0].shape[1] if all_fbonds else 37
   +    padding  = torch.zeros(1, bond_dim)
   +    padding_bg = torch.zeros(1, 10, dtype=torch.long)
   +
   +    mol_batch = (
   +        torch.cat(all_fatoms,     dim=0),
   ... (+303 satir daha)

--- PLANET_datautils_esm2_vector_smooth.py : 160 satir | +72 -92 (base'e gore)
   @@ -6,2 +6,2 @@
   -# Kendi modüllerin (Aynı dizinde olduğu varsayılıyor)
   -from chemutils import ComplexPocket, tensorize_all
   +# ESM2 uyumlu modülün
   +from chemutils_esm2_vector_smooth import tensorize_all 
   @@ -9,2 +9,3 @@
   -# JSON dosyasının yolu
   -PK_MAP_PATH_LOCAL = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
   +# --- KESİN YOLLAR ---
   +PK_MAP_JSON = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
   +ESM_SUBDIR = "planet_esm2_pkl"
   @@ -14 +15 @@
   -        # 1. ⚡ YEREL SSD AYARI: os.getlogin() yerine getpass.getuser() (HPC Uyumu)
   +        # 1. ⚡ YEREL SSD AYARI (HPC Uyumlu)
   @@ -17 +18 @@
   -        
   +
   @@ -19 +20 @@
   -            print(f"UYARI: {data_pickle_path} bulunamadı!")
   +            print(f"⚠️ {data_pickle_path} bulunamadı!")
   @@ -28 +29 @@
   -        # 2. ⚖️ DENGELEME MANTIĞI: Binder ve Decoy dengeli batch'ler oluşturur
   +        # 2. ⚖️ DENGELEME MANTIĞI
   @@ -33,21 +34,15 @@
   -            if len(binders) == 0:
   -                self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]
   -            else:
   -                decoys_per_batch = max(1, batch_size - binder_per_batch)
   -                num_batches = max(1, len(binders) // binder_per_batch)
   -                
   -                random.shuffle(binders)
   -                random.shuffle(decoys)
   -                
   -                from itertools import cycle
   -                decoy_cycle = cycle(decoys) if decoys else []
   -                
   -                balanced_batches = []
   -                for i in range(num_batches):
   -                    batch = binders[i*binder_per_batch : (i+1)*binder_per_batch]
   -                    if decoy_cycle:
   -                        batch += [next(decoy_cycle) for _ in range(decoys_per_batch)]
   -                    random.shuffle(batch)
   -                    balanced_batches.append(batch)
   -                self.data = balanced_batches
   -                print(f"✅ DATASET YÜKLENDİ: {len(self.data)} dengeli batch hazır.")
   +            num_batches = len(binders) // binder_per_batch if binders else 1
   +            random.shuffle(binders)
   +            
   +            from itertools import cycle
   +            decoy_cycle = cycle(decoys) if decoys else []
   +
   +            balanced_batches = []
   +            for i in range(num_batches):
   +                batch_binders = binders[i * binder_per_batch : (i + 1) * binder_per_batch]
   +                batch_decoys = [next(decoy_cycle) for _ in range(batch_size - binder_per_batch)] if decoys else []
   +                batch = batch_binders + batch_decoys
   +                random.shuffle(batch)
   +                balanced_batches.append(batch)
   +            self.data = balanced_batches
   +            print(f"✅ ESM2 DATASET HAZIR: {len(self.data)} Batch.")
   @@ -55 +50 @@
   -            self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]
   +            self.data = [full_data[i : i + batch_size] for i in range(0, len(full_data), batch_size)]
   @@ -57,10 +52 @@
   -    def __len__(self):
   -        return len(self.data)
   -
   -    def __getitem__(self, idx):
   -        try:
   -            return self.tensorize(idx)
   -        except Exception as e:
   -            # Hata durumunda batch'i atla ama logla
   -            # print(f"VERİ HATASI: {e}", file=sys.stderr)
   -            return None
   +    def __len__(self): return len(self.data)
   @@ -70 +55,0 @@
   -        
   @@ -72 +57 @@
   -            source_path = record[0] 
   +            source_path, pk_val = record[0], record[1]
   @@ -76,3 +61,3 @@
   -            # 3. 📂 BYTE OPTIMIZASYONU VE SSD TRANSFERİ
   -            if is_multi and "decoys.pkl" in file_name:
   -                byte_file_name = file_name.replace("_decoys.pkl", "_decoys_byte.pkl")
   +            # 3. 📂 ESM2 BYTE-OFFSET OPTIMIZASYONU
   +            if is_multi and "_esm512d.pkl" in source_path:
   +                byte_file_name = file_name.replace("_esm512d.pkl", "_byte_esm512d.pkl")
   @@ -86 +71 @@
   -            # --- 🛡️ AKILLI DISK KONTROLÜ (EN KRİTİK KISIM) ---
   +            # --- 🛡️ AKILLI DISK KONTROLÜ (10 GB EMNİYET KİLİDİ) ---
   ... (+107 satir daha)

==========================================================================================
GROUP: Dataset_preparation/ CHEMUTILS (base=chemutils.py)
==========================================================================================
BASE Dataset_preparation/chemutils.py: 456 satir

--- chemutils_esm2.py : 450 satir | +62 -68 (base'e gore)
   @@ -17,0 +18 @@
   +# --- AYARLAR ---
   @@ -19,0 +21,2 @@
   +#FEATURE_DIM = 512  # <--- ESM-2 İÇİN GÜNCELLENDİ (Eskiden 20 idi)
   +FEATURE_DIM = 512
   @@ -20,0 +24,2 @@
   +
   +# BLOSUM62 (Yedek olarak duruyor, ESM-2 varsa kullanılmaz)
   @@ -95,2 +99,0 @@
   -        
   -        # --- 1. LİGANDI OKU VE CENTERIOD'U HESAPLA ---
   @@ -98,18 +101,3 @@
   -            mol_rdkit = None
   -            try:
   -                if ligand_sdf.lower().endswith('.mol2'):
   -                    mol_rdkit = Chem.MolFromMol2File(ligand_sdf, sanitize=False, removeHs=False)
   -                elif ligand_sdf.lower().endswith('.sdf'):
   -                    mol_rdkit = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)[0]
   -                
   -                if mol_rdkit is None:
   -                    raise ValueError("RDKit geçerli bir molekül objesi okuyamadı.") 
   -                
   -                # Mol sınıfınızın init'i NoneType hatasını yakalar (daha önceki düzeltmeniz)
   -                self.ligand = Mol(mol_rdkit) 
   -                centeroid = self.ligand.compute_centeroid()
   -            
   -            except Exception as e:
   -                # Okuma veya Mol.__init__ başarısız olursa
   -                raise ValueError(f"Ligand okunamadı/merkez hesaplanamadı: {e}")
   -
   +            mol_rdkit = Chem.SDMolSupplier(ligand_sdf,sanitize=False,removeHs=False)[0]
   +            self.ligand = Mol(mol_rdkit) 
   +            centeroid = self.ligand.compute_centeroid()
   @@ -117 +104,0 @@
   -            # Koordinatlar doğrudan verilmişse
   @@ -120,5 +106,0 @@
   -        # --- 2. GEREKLİ CENTERIOD KONTROLÜ (Başarısız olursa hata fırlatır) ---
   -        if centeroid is None:
   -            raise ValueError("Pocket merkez koordinatları (centeroid) hesaplanamadı veya verilmedi.")
   -
   -        # --- 3. CEP BELİRLEME (Centeroid'in dolu olduğu garanti) ---
   @@ -126,7 +107,0 @@
   -
   -        # --- 4. SONUÇLARIN BİRLEŞTİRİLMESİ VE KRİTİK KONTROLÜ ---
   -        if not self.pocket_residues:
   -            # Hata mesajı artık bu ValueError ile atlatılacak.
   -            raise ValueError("Pocket içerisinde 12Å mesafede geçerli amino asit kalıntısı bulunamadı.")
   -            
   -        # Eğer bu noktaya gelindiyse, pocket_residues DOLUDUR. Concatenate güvenlidir.
   @@ -140,3 +115 @@
   -            # HETATM dahil tüm ATOM satırlarını oku
   -            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM') or line.startswith('HETATM')] 
   -            
   +            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')] 
   @@ -145 +117,0 @@
   -        
   @@ -149 +120,0 @@
   -            
   @@ -151 +121,0 @@
   -        
   @@ -155,11 +125,3 @@
   -                
   -                # 🌟 KESİN ÇÖZÜM: RESIDUE SINIFINA 3 ARGÜMANI DA İLET 🌟
   -                self.pocket_residues.append(Residue(
   -                    residue_content, 
   -                    res_num_str=index, 
   -                    chain_id=chain
   -                ))
   -            except Exception as e:
   -                # Residue sınıfı içindeki hatayı yakala (örneğin BLOSUM62 sözlüğü hatası)
   -                # print(f"Residue {index} {chain} işlenirken hata: {e}") 
   -                continue # Hatalı kalıntıyı atla ve devam et
   +                self.pocket_residues.append(Residue(residue_content, res_num_str=index, chain_id=chain))
   +            except:
   +                continue
   @@ -190 +151,0 @@
   -        # Düzeltme burada: index ve chain bilgisi Residue kurucusuna iletiliyor
   @@ -194,2 +154,0 @@
   -                
   -                # KRİTİK DÜZELTME: res_num_str=index, chain_id=chain parametrelerini ekledik
   @@ -200,0 +160,5 @@
   +        
   +        # --- ESM-2 GÜNCELLEMESİ İÇİN KRİTİK NOT ---
   +        # Burada orijinalde feature'lar oluşturuluyor. Ancak PKL dosyasını yüklerken
   +        # bu class tekrar çalıştırılmaz, pickle.load direkt veriyi getirir.
   +        # Bu __init__ sadece sıfırdan oluşturulurken çalışır.
   @@ -234 +198 @@
   -# --- chemutils.py içindeki Residue Sınıfı (GÜNCELLENDİ) ---
   +
   @@ -236,2 +200 @@
   ... (+74 satir daha)

--- chemutils_esm2_vector.py : 464 satir | +93 -85 (base'e gore)
   @@ -17,0 +18 @@
   +# --- AYARLAR ---
   @@ -19,0 +21 @@
   +FEATURE_DIM = 512
   @@ -20,0 +23,2 @@
   +
   +# BLOSUM62 (Yedek)
   @@ -95,2 +98,0 @@
   -        
   -        # --- 1. LİGANDI OKU VE CENTERIOD'U HESAPLA ---
   @@ -98,18 +100,3 @@
   -            mol_rdkit = None
   -            try:
   -                if ligand_sdf.lower().endswith('.mol2'):
   -                    mol_rdkit = Chem.MolFromMol2File(ligand_sdf, sanitize=False, removeHs=False)
   -                elif ligand_sdf.lower().endswith('.sdf'):
   -                    mol_rdkit = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)[0]
   -                
   -                if mol_rdkit is None:
   -                    raise ValueError("RDKit geçerli bir molekül objesi okuyamadı.") 
   -                
   -                # Mol sınıfınızın init'i NoneType hatasını yakalar (daha önceki düzeltmeniz)
   -                self.ligand = Mol(mol_rdkit) 
   -                centeroid = self.ligand.compute_centeroid()
   -            
   -            except Exception as e:
   -                # Okuma veya Mol.__init__ başarısız olursa
   -                raise ValueError(f"Ligand okunamadı/merkez hesaplanamadı: {e}")
   -
   +            mol_rdkit = Chem.SDMolSupplier(ligand_sdf,sanitize=False,removeHs=False)[0]
   +            self.ligand = Mol(mol_rdkit) 
   +            centeroid = self.ligand.compute_centeroid()
   @@ -117 +103,0 @@
   -            # Koordinatlar doğrudan verilmişse
   @@ -120,5 +105,0 @@
   -        # --- 2. GEREKLİ CENTERIOD KONTROLÜ (Başarısız olursa hata fırlatır) ---
   -        if centeroid is None:
   -            raise ValueError("Pocket merkez koordinatları (centeroid) hesaplanamadı veya verilmedi.")
   -
   -        # --- 3. CEP BELİRLEME (Centeroid'in dolu olduğu garanti) ---
   @@ -126,7 +106,0 @@
   -
   -        # --- 4. SONUÇLARIN BİRLEŞTİRİLMESİ VE KRİTİK KONTROLÜ ---
   -        if not self.pocket_residues:
   -            # Hata mesajı artık bu ValueError ile atlatılacak.
   -            raise ValueError("Pocket içerisinde 12Å mesafede geçerli amino asit kalıntısı bulunamadı.")
   -            
   -        # Eğer bu noktaya gelindiyse, pocket_residues DOLUDUR. Concatenate güvenlidir.
   @@ -140,3 +114 @@
   -            # HETATM dahil tüm ATOM satırlarını oku
   -            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM') or line.startswith('HETATM')] 
   -            
   +            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')] 
   @@ -145 +116,0 @@
   -        
   @@ -149 +119,0 @@
   -            
   @@ -151 +120,0 @@
   -        
   @@ -155,11 +124,3 @@
   -                
   -                # 🌟 KESİN ÇÖZÜM: RESIDUE SINIFINA 3 ARGÜMANI DA İLET 🌟
   -                self.pocket_residues.append(Residue(
   -                    residue_content, 
   -                    res_num_str=index, 
   -                    chain_id=chain
   -                ))
   -            except Exception as e:
   -                # Residue sınıfı içindeki hatayı yakala (örneğin BLOSUM62 sözlüğü hatası)
   -                # print(f"Residue {index} {chain} işlenirken hata: {e}") 
   -                continue # Hatalı kalıntıyı atla ve devam et
   +                self.pocket_residues.append(Residue(residue_content, res_num_str=index, chain_id=chain))
   +            except:
   +                continue
   @@ -190 +150,0 @@
   -        # Düzeltme burada: index ve chain bilgisi Residue kurucusuna iletiliyor
   @@ -194,2 +153,0 @@
   -                
   -                # KRİTİK DÜZELTME: res_num_str=index, chain_id=chain parametrelerini ekledik
   @@ -200,0 +159 @@
   +        
   @@ -234 +193 @@
   -# --- chemutils.py içindeki Residue Sınıfı (GÜNCELLENDİ) ---
   +
   @@ -236,2 +195 @@
   -    # res_num_str ve chain_id parametreleri eklendi!
   -    def __init__(self, residue_content, res_num_str, chain_id): 
   +    def __init__(self, residue_content, res_num_str=None, chain_id=None):
   @@ -239,5 +196,0 @@
   -        
   ... (+136 satir daha)

--- chemutils_esm2_vector_smooth.py : 500 satir | +148 -104 (base'e gore)
   @@ -17,0 +18 @@
   +# --- AYARLAR ---
   @@ -19,0 +21 @@
   +FEATURE_DIM = 512
   @@ -20,0 +23,2 @@
   +
   +# BLOSUM62 (Yedek)
   @@ -49,0 +54,8 @@
   +# --- [YENİ] Gaussian Decay Fonksiyonu ---
   +def gaussian_decay(distance, sigma=4.0):
   +    """
   +    Mesafeye bağlı yumuşak etiket üretir.
   +    Sert 4.0 Angstrom eşiği yerine, sürekli azalan bir skor verir.
   +    """
   +    return np.exp(- np.square(distance) / np.square(sigma))
   +
   @@ -95,2 +106,0 @@
   -        
   -        # --- 1. LİGANDI OKU VE CENTERIOD'U HESAPLA ---
   @@ -98,18 +108,3 @@
   -            mol_rdkit = None
   -            try:
   -                if ligand_sdf.lower().endswith('.mol2'):
   -                    mol_rdkit = Chem.MolFromMol2File(ligand_sdf, sanitize=False, removeHs=False)
   -                elif ligand_sdf.lower().endswith('.sdf'):
   -                    mol_rdkit = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)[0]
   -                
   -                if mol_rdkit is None:
   -                    raise ValueError("RDKit geçerli bir molekül objesi okuyamadı.") 
   -                
   -                # Mol sınıfınızın init'i NoneType hatasını yakalar (daha önceki düzeltmeniz)
   -                self.ligand = Mol(mol_rdkit) 
   -                centeroid = self.ligand.compute_centeroid()
   -            
   -            except Exception as e:
   -                # Okuma veya Mol.__init__ başarısız olursa
   -                raise ValueError(f"Ligand okunamadı/merkez hesaplanamadı: {e}")
   -
   +            mol_rdkit = Chem.SDMolSupplier(ligand_sdf,sanitize=False,removeHs=False)[0]
   +            self.ligand = Mol(mol_rdkit) 
   +            centeroid = self.ligand.compute_centeroid()
   @@ -117 +111,0 @@
   -            # Koordinatlar doğrudan verilmişse
   @@ -120,5 +113,0 @@
   -        # --- 2. GEREKLİ CENTERIOD KONTROLÜ (Başarısız olursa hata fırlatır) ---
   -        if centeroid is None:
   -            raise ValueError("Pocket merkez koordinatları (centeroid) hesaplanamadı veya verilmedi.")
   -
   -        # --- 3. CEP BELİRLEME (Centeroid'in dolu olduğu garanti) ---
   @@ -126,7 +114,0 @@
   -
   -        # --- 4. SONUÇLARIN BİRLEŞTİRİLMESİ VE KRİTİK KONTROLÜ ---
   -        if not self.pocket_residues:
   -            # Hata mesajı artık bu ValueError ile atlatılacak.
   -            raise ValueError("Pocket içerisinde 12Å mesafede geçerli amino asit kalıntısı bulunamadı.")
   -            
   -        # Eğer bu noktaya gelindiyse, pocket_residues DOLUDUR. Concatenate güvenlidir.
   @@ -140,3 +122 @@
   -            # HETATM dahil tüm ATOM satırlarını oku
   -            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM') or line.startswith('HETATM')] 
   -            
   +            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')] 
   @@ -145 +124,0 @@
   -        
   @@ -149 +127,0 @@
   -            
   @@ -151 +128,0 @@
   -        
   @@ -155,11 +132,3 @@
   -                
   -                # 🌟 KESİN ÇÖZÜM: RESIDUE SINIFINA 3 ARGÜMANI DA İLET 🌟
   -                self.pocket_residues.append(Residue(
   -                    residue_content, 
   -                    res_num_str=index, 
   -                    chain_id=chain
   -                ))
   -            except Exception as e:
   -                # Residue sınıfı içindeki hatayı yakala (örneğin BLOSUM62 sözlüğü hatası)
   -                # print(f"Residue {index} {chain} işlenirken hata: {e}") 
   -                continue # Hatalı kalıntıyı atla ve devam et
   +                self.pocket_residues.append(Residue(residue_content, res_num_str=index, chain_id=chain))
   +            except:
   +                continue
   @@ -190 +158,0 @@
   -        # Düzeltme burada: index ve chain bilgisi Residue kurucusuna iletiliyor
   @@ -194,2 +161,0 @@
   -                
   -                # KRİTİK DÜZELTME: res_num_str=index, chain_id=chain parametrelerini ekledik
   @@ -199,0 +166,2 @@
   +        # Dikkat: Bu sınıf PKL'den yüklendiğinde __init__ çalışmaz. 
   ... (+220 satir daha)

--- chemutils_dude.py : 229 satir | +127 -354 (base'e gore)
   @@ -1,4 +1,2 @@
   -from operator import le
   -from numpy.core.fromnumeric import shape
   -import pandas as pd
   -import rdkit,os
   +
   +print("DEBUG: chemutils_dude dosyası yükleniyor... (VERSION: FIXED_RESTORED)", flush=True)
   @@ -6,8 +3,0 @@
   -from rdkit.Chem import AllChem
   -from rdkit.Chem import Descriptors
   -from rdkit.Chem.QED import qed
   -from scipy.sparse import coo, csr_matrix
   -from scipy.sparse.csgraph import minimum_spanning_tree
   -from collections import defaultdict
   -from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
   -#from vocab import Vocab
   @@ -16,4 +6,4 @@
   -from typing import List,Tuple
   -
   -MST_MAX_WEIGHT = 100 
   -MAX_NCAND = 2000
   +from collections import defaultdict
   +
   +# --- SABITLER ---
   +MAX_NB = 10
   @@ -42,0 +33 @@
   +
   @@ -45,2 +36 @@
   -
   -ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1  #31
   +ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1 
   @@ -48 +37,0 @@
   -MAX_NB = 10
   @@ -51,2 +40 @@
   -    if x not in allowable_set:
   -        x = allowable_set[-1]
   +    if x not in allowable_set: x = allowable_set[-1]
   @@ -75,4 +63 @@
   -    if distance > 12.0 :
   -        return False
   -    else:
   -        return True
   +    return distance <= 12.0
   @@ -81,0 +67,2 @@
   +    if not pdb_lines: return np.zeros((1,3)), np.zeros((0,3)), np.zeros((0,3))
   +
   @@ -83,84 +70,26 @@
   -    coordinates = np.array([
   -        [float(line[30:38]),float(line[38:46]),float(line[46:54])] 
   -        for line in pdb_lines],dtype=np.float32)
   -    mass_center = np.sum(coordinates * atoms_mass,axis=0,keepdims=True) / np.sum(atoms_mass)
   -    alpha_position =  np.array([
   -        [float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pdb_lines if line[12:16].strip()=="CA"],dtype=np.float32)
   -
   -    return mass_center,coordinates,alpha_position
   -
   -class ProteinPocket():
   -    def __init__(self,protein_pdb,centeriod_x=None,centeriod_y=None,centeriod_z=None,ligand_sdf=None):
   -        centeroid = None
   -        
   -        # --- 1. LİGANDI OKU VE CENTERIOD'U HESAPLA ---
   -        if ligand_sdf is not None:
   -            mol_rdkit = None
   -            try:
   -                if ligand_sdf.lower().endswith('.mol2'):
   -                    mol_rdkit = Chem.MolFromMol2File(ligand_sdf, sanitize=False, removeHs=False)
   -                elif ligand_sdf.lower().endswith('.sdf'):
   -                    mol_rdkit = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)[0]
   -                
   -                if mol_rdkit is None:
   -                    raise ValueError("RDKit geçerli bir molekül objesi okuyamadı.") 
   -                
   -                # Mol sınıfınızın init'i NoneType hatasını yakalar (daha önceki düzeltmeniz)
   -                self.ligand = Mol(mol_rdkit) 
   -                centeroid = self.ligand.compute_centeroid()
   -            
   -            except Exception as e:
   -                # Okuma veya Mol.__init__ başarısız olursa
   -                raise ValueError(f"Ligand okunamadı/merkez hesaplanamadı: {e}")
   -
   -        elif centeriod_x is not None and centeriod_y is not None and centeriod_z is not None:
   -            # Koordinatlar doğrudan verilmişse
   -            centeroid = np.array([centeriod_x,centeriod_y,centeriod_z],dtype=np.float32)
   -        
   -        # --- 2. GEREKLİ CENTERIOD KONTROLÜ (Başarısız olursa hata fırlatır) ---
   -        if centeroid is None:
   -            raise ValueError("Pocket merkez koordinatları (centeroid) hesaplanamadı veya verilmedi.")
   -
   -        # --- 3. CEP BELİRLEME (Centeroid'in dolu olduğu garanti) ---
   -        self.determine_pocket_residues(protein_pdb,centeroid)
   ... (+424 satir daha)
```
