#!/bin/bash
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20   
#SBATCH --job-name=planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_UW_1:11"
SSD_DIR="/tmp/hnaydin_lmdb"

mkdir -p $SAVE_DIR
mkdir -p $SSD_DIR

echo "--- SSD Durumu ---"
df -h /tmp

echo "--- LMDB SSD'ye Kopyalanıyor: $(date) ---"
cp -r $DATA_ROOT/train.lmdb $SSD_DIR/
cp    $DATA_ROOT/train_meta.json $SSD_DIR/
cp -r $DATA_ROOT/valid.lmdb $SSD_DIR/
cp    $DATA_ROOT/valid_meta.json $SSD_DIR/
cp -r $DATA_ROOT/core.lmdb $SSD_DIR/
cp    $DATA_ROOT/core_meta.json $SSD_DIR/
echo "--- Kopyalama Bitti: $(date) ---"
df -h /tmp

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "--- Eğitim Başlıyor: $(date) ---"
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_train_esm2.py \
    -t $SSD_DIR/train.lmdb \
    -v $SSD_DIR/valid.lmdb \
    -te $SSD_DIR/core.lmdb \
    -d $SAVE_DIR \
    --decoy_count 11 \
    --epoch 100 \
    --anneal_iter 20000 \
    --save_iter 20000 \
    --print_iter 1000 \
    --initial_step 1440000 \
    --load_checkpoint $SAVE_DIR/PLANET.iter-1440000 \
    --lr 0.00002378

rm -rf $SSD_DIR
echo "--- Temizlik Tamamlandı ---"
echo "İş Bitti: $(date)"