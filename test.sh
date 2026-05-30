#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --job-name=DEBUG_TEST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:05:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/debug_test_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/debug_test_%j.err
#SBATCH -A hnaydin

echo "----------------------------------------"
echo "✅ SLURM ÇALIŞIYOR!"
echo "Tarih: $(date)"
echo "Sunucu: $(hostname)"
echo "Dosya Sistemi Testi:"
ls -ld /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/
echo "----------------------------------------"
