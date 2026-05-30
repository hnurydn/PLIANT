#!/bin/bash
# --- PLANET_train_gpu.sh ---
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20   
#SBATCH --job-name=lr00005_planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.err
#SBATCH -A hnaydin

# 1. Ortamı Hazırla
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 2. Kayıt Dizini
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL"
# SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_noESM_noUW_2"
mkdir -p $SAVE_DIR

echo "--- SSD Durumu (Baslangic) ---"
df -h /tmp
echo "------------------------------"

# 2. 🛡️ ÖN TEMİZLİK (En Önemli Kısım)
# Eğer bir önceki işten kalıntı varsa, eğitime başlamadan masayı süpürelim.
echo "--- İş Başlıyor: $(date) ---"
rm -rf /tmp/hnaydin_planet_cache
mkdir -p /tmp/hnaydin_planet_cache

# 3. Eğitimi Başlat (Adresler zaten kalıcı olarak güncellendi!)
# python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train_noESM_noUW.py \
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train2.py \
  -t /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index/train.pkl \
  -v /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index/valid.pkl \
  -te /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index/core.pkl \
  -d $SAVE_DIR \
  --lr 1e-5 \
  --batch_size 24 \
  --epoch 250 

rm -rf /tmp/hnaydin_planet_cache
echo "Temizlik tamamlandi, disk kontrol ediliyor:"
df -h /tmp