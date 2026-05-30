#!/bin/bash
# --- PLANET_train_gpu.sh ---
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20  
#SBATCH --job-name=lr00005_planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.err
#SBATCH -A hnaydin

echo "Is basladi (GPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL"
mkdir -p $SAVE_DIR

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train_metrics.py \
-t $DATA_ROOT/train.pkl \
-v $DATA_ROOT/valid.pkl \
-te $DATA_ROOT/core.pkl \
-d $SAVE_DIR \
--lr 1e-5 \
--batch_size 24 \
--epoch 100 

echo "Is bitti (CPU OPTIMIZE TEST): $(date)"