#!/bin/bash
# --- launch_baseline_ligandlm.sh (TRUBA) ---
#SBATCH -C weka
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -J baseline_ligandlm
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/baseline_ligandlm_%j.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/baseline_ligandlm_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model

echo "Is basladi: $(date)"; echo "Host: $(hostname)"
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v3
SAVE=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/noESM_noUW_LigandLM

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_train_baseline_ligandlm.py \
    -t $DATA/train.lmdb -v $DATA/valid.lmdb -te $DATA/core.lmdb \
    -d $SAVE \
    --seed 42 --decoy_count 10

echo "Is bitti: $(date)"
