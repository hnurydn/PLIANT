#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=create_lmdb
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/create_lmdb_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/create_lmdb_%j.err
#SBATCH -A hnaydin


echo "Is basladi: $(date)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen


python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/update_lmdb_decoys.py \
    --lmdb_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet \
    --sdf_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/decoys_sdf_3 \
    --sets train valid core

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/update_lmdb_decoys.py \
    --lmdb_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2 \
    --sdf_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/decoys_sdf_3 \
    --sets train valid core

echo "Is bitti: $(date)"

# python3 /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/create_lmdb.py \
   # --src_lmdb /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet \
   # --out_lmdb /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2 \
    #--sets train valid core