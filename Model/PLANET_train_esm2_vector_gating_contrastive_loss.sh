#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=planet_con_1b11d
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/train_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
CKPT_SOURCE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_VG_CONTRASTIVE_1b11d_2/PLANET.iter-1900000"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_VG_CONTRASTIVE_v3_finetune"
SSD_DIR="/tmp/hnaydin_v3_ft_${SLURM_JOB_ID}"

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

echo "--- Fine-tune Başlıyor: $(date) ---"
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_train_esm2_vector_gating_contrastive_loss.py \
    -t  $SSD_DIR/train.lmdb \
    -v  $SSD_DIR/valid.lmdb \
    -te $SSD_DIR/core.lmdb \
    -d  $SAVE_DIR \
    --decoy_count 11 \
    --epoch 10 \
    --anneal_iter 20000 \
    --save_iter 10000 \
    --print_iter 1000 \
    --initial_step 1900000 \
    --load_checkpoint $CKPT_SOURCE \
    --lr 0.00001424

rm -rf $SSD_DIR
echo "--- Temizlik Tamamlandı ---"
echo "İş Bitti: $(date)"