#!/bin/bash
# --- wandb_merge_all.sh ---
#SBATCH -C weka
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=20
#SBATCH --job-name=planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/wandb_merge_all_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/wandb_merge_all_%j.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model

echo "Is basladi (CPU OPTIMIZE TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/wandb_merge_all.py
