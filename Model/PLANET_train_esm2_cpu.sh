#!/bin/bash
# --- PLANET_train_gpu.sh ---
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54 
#SBATCH --job-name=planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.err
#SBATCH -A hnaydin

echo "Is basladi (CPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM_1:1"
mkdir -p $SAVE_DIR

echo "--- SSD Durumu (Baslangic) ---"
df -h /tmp
echo "------------------------------"

# 2. 🛡️ ÖN TEMİZLİK (En Önemli Kısım)
# Eğer bir önceki işten kalıntı varsa, eğitime başlamadan masayı süpürelim.
echo "--- İş Başlıyor: $(date) ---"
rm -rf /tmp/hnaydin_planet_cache
mkdir -p /tmp/hnaydin_planet_cache

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train_esm2.py \
    -t $DATA_ROOT/train_esm2.pkl \
    -v $DATA_ROOT/valid_esm2.pkl \
    -te $DATA_ROOT/core_esm2.pkl \
    -d $SAVE_DIR \
    --feature_dims 512 \
    --batch_size 16 \
    --lr  2e-5 \
    --clip_norm 1.0 \
    --epoch 1000 

rm -rf /tmp/hnaydin_planet_cache
echo "Temizlik tamamlandi, disk kontrol ediliyor:"
df -h /tmp

echo "Is bitti (CPU OPTIMIZE TEST): $(date)"
