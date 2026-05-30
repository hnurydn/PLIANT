#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --job-name=pretensorize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pretensorize_%A_%a.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pretensorize_%A_%a.err
#SBATCH --array=0-2
#SBATCH -A hnaydin

LMDB_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet"
OUT_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor"
mkdir -p $OUT_BASE

/arf/home/hnaydin/miniconda3/envs/pocketgen/bin/python \
    /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/pretensorize_lmdb.py \
    --input  /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet/valid.lmdb \
    --output /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/valid_tensor.lmdb \
    --njobs  40
    
echo "Is bitti: $(date) | Set: $SET_NAME"
