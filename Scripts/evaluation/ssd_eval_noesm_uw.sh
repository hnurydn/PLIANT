#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=planet_con_1b11d
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/train_%j.err
#SBATCH -A hnaydin


echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet"
SSD_DIR="/tmp/hnaydin_eval_${SLURM_JOB_ID}_2"
mkdir -p $SSD_DIR

echo "--- Gerekli veriler SSD'ye kopyalanıyor ---"
cp -r $DATA_ROOT/valid.lmdb $SSD_DIR/
cp    $DATA_ROOT/valid_meta.json $SSD_DIR/
cp -r $DATA_ROOT/core.lmdb $SSD_DIR/
cp    $DATA_ROOT/core_meta.json $SSD_DIR/

DIMS=300  # ESM'siz modelse 300 yaparsın
# -----------------------------------------------------------

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_re-evaluate_noesm.py \
    --ckpt_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/noESM_UW_1:11 \
    --valid_data $SSD_DIR/valid.lmdb \
    --test_data $SSD_DIR/core.lmdb \
    --run_name noesm_uw_1b11d \
    --dims $DIMS 

rm -rf $SSD_DIR