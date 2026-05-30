#!/bin/bash
#SBATCH -p barbun
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A hnaydin
#SBATCH -J 05_generate_decoy_candidates
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_decoy_candidate_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_decoy_candidate_%A.err
#SBATCH --time=04:00:00
#SBATCH --mem=60G


echo "Is basladi: $(date)"
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/05_generate_decoy_candidates.py

echo "Is bitti: $(date)"
