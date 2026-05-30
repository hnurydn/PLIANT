# lmdb_planet_v2 — PDBbind2020 (ESM2'li, v2)

Bu dizin PLANET modelinin **v2 (ESM2 embedding'li)** eğitim/doğrulama/test veri setine aittir.

## Bu repoda olanlar

Yalnızca küçük **meta** dosyaları versiyonlanır:

| Dosya | İçerik |
|-------|--------|
| `train_meta.json` | Eğitim seti kayıt indeksi / metadata |
| `valid_meta.json` | Doğrulama seti metadata |
| `core_meta.json`  | Core (test) seti metadata |

## Bu repoda OLMAYANLAR (boyut nedeniyle)

Asıl LMDB veri dosyaları **~25 GB** olduğu için Git'e dahil edilmemiştir
(GitHub dosya başına 100 MB sınırı). `.gitignore` bunları hariç tutar:

| Dosya | Boyut |
|-------|-------|
| `train.lmdb/data.mdb` | ~21.8 GB |
| `valid.lmdb/data.mdb` | ~2.65 GB |
| `core.lmdb/data.mdb`  | ~204 MB |

## Veriye nasıl ulaşılır

Tam LMDB veri seti TRUBA / ARF HPC kümesinde durmaktadır:

```
/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2/
├── core.lmdb/   (core.lmdb/data.mdb, lock.mdb)
├── train.lmdb/
└── valid.lmdb/
```

> **TODO:** Veri seti Zenodo veya Hugging Face'e yüklendiğinde indirme linki/DOI
> buraya eklenecek.

### Beklenen dizin yapısı (veriyi indirdikten sonra)

```
lmdb_planet_v2/
├── core.lmdb/data.mdb
├── core_meta.json
├── train.lmdb/data.mdb
├── train_meta.json
├── valid.lmdb/data.mdb
└── valid_meta.json
```
