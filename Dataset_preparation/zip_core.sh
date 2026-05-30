#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --job-name=zip_core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-12:00:00
#SBATCH --output=/arf/scratch/hnaydin/zip_%j.out
#SBATCH --error=/arf/scratch/hnaydin/zip_%j.err
#SBATCH -A hnaydin

echo "Is basladi: $(date)"

/arf/home/hnaydin/miniconda3/envs/pocketgen/bin/python  /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/convert.py

cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet/
tar --sparse -czf core_lmdb.tar.gz core.lmdb core_meta.json

echo "Is bitti: $(date)"
