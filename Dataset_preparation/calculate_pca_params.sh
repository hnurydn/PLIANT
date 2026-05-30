#!/bin/bash
# --- sıkıstırma_job.sh ---
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --job-name=veri_sikistirma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G  # Sıkıştırma için 100G aşırı olabilir, 5-10G yeterli olmalı.
#SBATCH --time=12:00:00 # İşin 12 saate kadar sürmesine izin verin (3 gün gereksiz)
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/sikistirma_cikti_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/sikistirma_hata_%j.err
#SBATCH -A hnaydin

echo "Is basladi (Adim 4 - Decoy Ekleme): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
# pocketgen environment'ını aktive et
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# Script'in bulunduğu ana dizine git (chemutils.py'yi bulabilmek için)
cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model

echo "Calistirilacak Python script'i: calculate_pca_params_dude.py"

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/calculate_pca_params_dude.py

echo "Is bitti (Adim 4): $(date)"