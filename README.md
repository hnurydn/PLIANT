# PLANET — Protein–Ligand Bağlanma Tahmini Modeli

Bu repo, yüksek lisans tezi kapsamında geliştirilen **PLANET** derin öğrenme modelinin
eğitim, veri hazırlama ve yardımcı scriptlerini içerir.

Model, protein–ligand bağlanma tahmini için farklı mimarileri karşılaştırır:
ESM2 embeddings + Uncertainty Weighting (UW) + Vector Gating + Contrastive Loss
(çeşitli versiyonlar).

> Kaynak: TRUBA / ARF HPC kümesi (ULAKBIM) üzerindeki
> `/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model` çalışma dizini.
> Bu repo yalnızca **kod** içerir; eğitilmiş model ağırlıkları (`*.safetensors` vb.),
> büyük veri dosyaları ve loglar `.gitignore` ile hariç tutulmuştur (sunucuda kalır).

## Dizin Yapısı

```
.
├── Dataset_preparation/     # Veri hazırlama / ön işleme scriptleri
├── Model/                   # Model tanımları ve eğitim scriptleri
├── Scripts/                 # Yardımcı scriptler
├── article_pose_dependence/ # Makale reprodüktibilite paketi (bkz. aşağı)
└── *.py, *.sh                # Üst seviye çalıştırma ve Slurm scriptleri
```

## Makale: Pose-(In)dependence Reprodüktibilite Paketi

Tez sonrası yapılan ek analizler (pose-perturbation deneyi, checkpoint
robustness, seed-robustness, DUD-E/LIT-PCBA istatistiksel anlamlılık testleri
ve figürler) — *Journal of Chemical Information and Modeling*'e gönderilen
makalenin kodu ve türetilmiş verileri:

📁 **[`article_pose_dependence/`](article_pose_dependence/)**

## Veri Setleri

Eğitim/test verileri (LMDB) büyük olduğu için (~34 GB) GitHub'da tutulmaz;
**Hugging Face**'te private bir dataset olarak barındırılır:

🤗 **https://huggingface.co/datasets/hnurydn/PLANET-datasets**

| Alt klasör | Açıklama | Boyut |
|-----------|----------|-------|
| `lmdb_planet/` | ESM'siz (no-ESM) — `noESM_*` modelleri | ~9 GB |
| `lmdb_planet_v2/` | ESM2 embedding'li (v2) | ~25 GB |

Klonlamak için (private olduğundan önce `huggingface-cli login` ile giriş gerekir):

```bash
git clone https://huggingface.co/datasets/hnurydn/PLANET-datasets
```

veya sadece bir alt klasörü indirmek için:

```bash
huggingface-cli download hnurydn/PLANET-datasets --repo-type dataset \
  --include "lmdb_planet_v2/*" --local-dir datasets/pdbbind2020
```

Detaylı bilgi: [`datasets/pdbbind2020/lmdb_planet/DATASET.md`](datasets/pdbbind2020/lmdb_planet/DATASET.md)
ve [`.../lmdb_planet_v2/DATASET.md`](datasets/pdbbind2020/lmdb_planet_v2/DATASET.md).

## Notlar

- İşler TRUBA ARF kümesinde **Slurm** ile çalıştırılır (`sbatch *.sh`).
- WandB projeleri: `PLANET_TRAINING` → `PLANET_COMBINED_RESULT`.
- Conda ortamı: `pocketgen`.
