# lmdb_planet — PDBbind2020 (ESM'siz / no-ESM)

Bu dizin PLANET modelinin **ESM'siz (no-ESM)** eğitim/doğrulama/test veri setine aittir.
`noESM_UW` ve `noESM_noUW` modelleri bu veri setiyle eğitilmiştir.
(ESM2 embedding'li versiyon için bkz. `../lmdb_planet_v2/`.)

## Bu repoda olanlar

Yalnızca küçük **meta** dosyaları versiyonlanır:

| Dosya | İçerik |
|-------|--------|
| `train_meta.json` | Eğitim seti kayıt indeksi (key, pK, is_decoy, pdb_id, n_decoys) |
| `valid_meta.json` | Doğrulama seti metadata |
| `core_meta.json`  | Core (test) seti metadata |

## Bu repoda OLMAYANLAR (boyut nedeniyle)

Asıl LMDB veri dosyaları **~9 GB** olduğu için Git'e dahil edilmemiştir
(GitHub dosya başına 100 MB sınırı). `.gitignore` bunları hariç tutar:

| Dosya | Boyut |
|-------|-------|
| `train.lmdb/data.mdb` | ~7.9 GB |
| `valid.lmdb/data.mdb` | ~993 MB |
| `core.lmdb/data.mdb`  | ~51 MB |

## Veriye nasıl ulaşılır

Tam LMDB veri seti TRUBA / ARF HPC kümesinde durmaktadır:

```
/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet/
├── core.lmdb/   (core.lmdb/data.mdb)
├── train.lmdb/
└── valid.lmdb/
```

> **TODO:** Veri seti Zenodo veya Hugging Face'e yüklendiğinde indirme linki/DOI
> buraya eklenecek.

### Beklenen dizin yapısı (veriyi indirdikten sonra)

```
lmdb_planet/
├── core.lmdb/data.mdb
├── core_meta.json
├── train.lmdb/data.mdb
├── train_meta.json
├── valid.lmdb/data.mdb
└── valid_meta.json
```
