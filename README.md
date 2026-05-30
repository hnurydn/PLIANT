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
├── Dataset_preparation/   # Veri hazırlama / ön işleme scriptleri
├── Model/                 # Model tanımları ve eğitim scriptleri
├── Scripts/               # Yardımcı scriptler
└── *.py, *.sh             # Üst seviye çalıştırma ve Slurm scriptleri
```

## Notlar

- İşler TRUBA ARF kümesinde **Slurm** ile çalıştırılır (`sbatch *.sh`).
- WandB projeleri: `PLANET_TRAINING` → `PLANET_COMBINED_RESULT`.
- Conda ortamı: `pocketgen`.
