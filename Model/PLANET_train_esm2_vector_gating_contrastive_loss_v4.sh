#!/bin/bash
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=planet_v4_proj
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/v4_proj_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/v4_proj_%j.err
#SBATCH -A hnaydin

set -euo pipefail

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

which python
python -V
nvidia-smi

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_VG_CONTRASTIVE_v4_projection"
SSD_DIR="/tmp/hnaydin_v4_proj_${SLURM_JOB_ID}"

mkdir -p $SAVE_DIR
mkdir -p $SSD_DIR

echo "--- SSD Durumu ---"
df -h /tmp

echo "--- LMDB SSD'ye Kopyalanıyor: $(date) ---"
cp -r $DATA_ROOT/train.lmdb  $SSD_DIR/
cp    $DATA_ROOT/train_meta.json $SSD_DIR/
cp -r $DATA_ROOT/valid.lmdb  $SSD_DIR/
cp    $DATA_ROOT/valid_meta.json $SSD_DIR/
cp -r $DATA_ROOT/core.lmdb   $SSD_DIR/
cp    $DATA_ROOT/core_meta.json  $SSD_DIR/
echo "--- Kopyalama Bitti: $(date) ---"
df -h /tmp

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "--- Eğitim Başlıyor: $(date) ---"
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_train_esm2_vector_gating_contrastive_loss_v4.py \
    -t  $SSD_DIR/train.lmdb \
    -v  $SSD_DIR/valid.lmdb \
    -te $SSD_DIR/core.lmdb \
    -d  $SAVE_DIR \
    --decoy_count 11 \
    --epoch 50 \
    --anneal_iter 20000 \
    --save_iter 20000 \
    --print_iter 1000 \
    --lr 0.00002503 \
    --load_checkpoint /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_VG_CONTRASTIVE_v4_projection/PLANET.iter-1380000 \
    --initial_step 1380000


rm -rf $SSD_DIR
echo "--- Temizlik Tamamlandı ---"
echo "İş Bitti: $(date)"