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
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_cpu_test_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_cpu_test_%j.err
#SBATCH -A hnaydin

echo "Is basladi (CPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/test_set_preparetion.py
