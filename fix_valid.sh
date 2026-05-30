#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --job-name=fix_valid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/fix_valid_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/fix_valid_%j.err
#SBATCH -A hnaydin

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/create_lmdb.py
