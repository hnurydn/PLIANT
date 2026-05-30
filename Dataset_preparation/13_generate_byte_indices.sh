#!/bin/bash
# --- 13_generate_byte_indices.sh---
# Amaç: PDBbind veri setindeki tüm PDB ID'leri için ComplexPocket PKL dosyalarını oluşturmak.

#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J 13_generate_byte_indices
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/13_generate_byte_indices_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/13_generate_byte_indices_%A.err
#SBATCH --time=1-00:00:00 
#SBATCH --mem=200G

# 1. ORTAM AYARI
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/13_generate_byte_indices.py