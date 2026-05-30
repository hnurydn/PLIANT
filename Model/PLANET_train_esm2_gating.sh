#!/bin/bash
# --- PLANET_train_esm2_gating.sh ---
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20   
#SBATCH --job-name=PLANET_train_esm2_gating
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_gating_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_gating_%j.err
#SBATCH -A hnaydin

echo "Is basladi (CPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_GATING"
mkdir -p $SAVE_DIR

echo "--- SSD Durumu (Baslangic) ---"
df -h /tmp
echo "------------------------------"

# 2. 🛡️ ÖN TEMİZLİK (En Önemli Kısım)
# Eğer bir önceki işten kalıntı varsa, eğitime başlamadan masayı süpürelim.
echo "--- İş Başlıyor: $(date) ---"
rm -rf /tmp/hnaydin_planet_cache
mkdir -p /tmp/hnaydin_planet_cache

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train_esm2_gating.py \
    -t $DATA_ROOT/train_esm2.pkl \
    -v $DATA_ROOT/valid_esm2.pkl \
    -te $DATA_ROOT/core_esm2.pkl \
    -d $SAVE_DIR \
    --feature_dims 512 \
    --batch_size 24 \
    --lr  1e-5 \
    --clip_norm 1.0 \
    --epoch 250 

rm -rf /tmp/hnaydin_planet_cache
echo "Temizlik tamamlandi, disk kontrol ediliyor:"
df -h /tmp

echo "Is bitti (CPU OPTIMIZE TEST): $(date)"
