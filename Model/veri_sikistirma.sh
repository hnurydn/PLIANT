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

# -----------------------------------------------------------
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/11_generate_test_fastas.py