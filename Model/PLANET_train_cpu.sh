#!/bin/bash
# --- PLANET_train_cpu.sh ---
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --job-name=lr00001_planet_cpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_cpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_cpu_train_%j.err
#SBATCH -A hnaydin

echo "Is basladi (CPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2"
mkdir -p $SAVE_DIR

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train_esm2.py \
-t $DATA_ROOT/train_esm2.pkl \
-v $DATA_ROOT/valid_esm2.pkl \
-te $DATA_ROOT/test_esm2.pkl \
-d $SAVE_DIR \
--feature_dims 512 \
--batch_size 24 \
--lr 0.00001 \
--clip_norm 1.0 \
--load_checkpoint $SAVE_DIR/PLANET_ESM2.iter-269865 \
--initial_step 269865

echo "Is bitti (CPU OPTIMIZE TEST): $(date)"