#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=PLANET_train_esm2_vector_finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_vector_finetune_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_vector_finetune_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
DATA_CROSSDOCKED="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/lmdb_crossdocked"
DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_VG_FINETUNE_NOAFF"
SSD_DIR="/tmp/hnaydin_vg_${SLURM_JOB_ID}"

mkdir -p $SAVE_DIR
mkdir -p $SSD_DIR

echo "--- SSD Durumu ---"
df -h /tmp

echo "--- LMDB SSD'ye Kopyalanıyor: $(date) ---"
cp -r $DATA_CROSSDOCKED/train.lmdb  $SSD_DIR/
cp    $DATA_CROSSDOCKED/train_meta_receptor.json $SSD_DIR/train_meta_receptor.json
cp    $DATA_CROSSDOCKED/train_meta_filtered.json $SSD_DIR/train_meta.json
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

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_finetune_crossdocked.py \
    --train $SSD_DIR/train.lmdb \
    --valid $SSD_DIR/valid.lmdb \
    --test  $SSD_DIR/core.lmdb \
    --save_dir $SAVE_DIR \
    --decoy_count 11 \
    --epoch 20 \
    --save_iter 20000 \
    --print_iter 1000 \
    --initial_step 1900000 \
    --load_checkpoint /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/ESM2_UW_VECTOR_GATING_1:11/PLANET.iter-1900000 \
    --lr 0.00001 \
    --run_name esm_vg_crossdocked_noaff \
    --freeze_affinity

rm -rf $SSD_DIR
echo "--- Temizlik Tamamlandı ---"
echo "İş Bitti: $(date)"
