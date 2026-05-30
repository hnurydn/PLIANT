#!/bin/bash
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=56 
#SBATCH --job-name=planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.err
#SBATCH -A hnaydin



echo "Is basladi: $(date)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen


# -p argümanı ile PDB ID'sini Python scriptine gönderiyoruz
python3 /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/10_update_pkl_esm2.py

echo "Is bitti: $(date)"
