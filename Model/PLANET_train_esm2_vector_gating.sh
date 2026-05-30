#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=PLANET_train_esm2_vector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_vector_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_vector_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_UW_VECTOR_GATING_1:11"
SSD_DIR="/tmp/hnaydin_vg_${SLURM_JOB_ID}"

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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "--- Eğitim Başlıyor: $(date) ---"
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_train_esm2_vector_gating.py \
    -t  $SSD_DIR/train.lmdb \
    -v  $SSD_DIR/valid.lmdb \
    -te $SSD_DIR/core.lmdb \
    -d  $SAVE_DIR \
    --decoy_count 11 \
    --epoch 50 \
    --anneal_iter 20000 \
    --save_iter 20000 \
    --print_iter 1000 \
    --initial_step 1420000 \
    --load_checkpoint /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_UW_VECTOR_GATING_1:11/PLANET.iter-1420000 \
    --lr 0.00002378

rm -rf $SSD_DIR
echo "--- Temizlik Tamamlandı ---"
echo "İş Bitti: $(date)"
