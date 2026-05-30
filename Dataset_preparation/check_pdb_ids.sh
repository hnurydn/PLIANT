#!/bin/bash
# --- PLANET_train_cpu.sh ---
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --job-name=lr00005_planet_cpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_cpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_cpu_train_%j.err
#SBATCH -A hnaydin

echo "Is basladi (CPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/check_pdb_ids.py > missing_pdb_ids.txt
