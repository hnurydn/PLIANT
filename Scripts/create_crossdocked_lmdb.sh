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


python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/create_crossdocked_lmdb.py \
    --pkl_dir       /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_pkls \
    --json_path     /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_new_records.json \
    --h5_path       /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pdbbind2020_esm2_embeddings_v2.h5 \
    --pca_path      /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/esm2_pca_params.pkl \
    --extracted_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_extracted \
    --out_lmdb      /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/lmdb_crossdocked \
    --map_size_gb   100