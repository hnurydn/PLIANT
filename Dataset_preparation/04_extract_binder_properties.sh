#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH -A hnaydin
#SBATCH -J 04_extract_binder_properties
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/extract_binder_properties_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/extract_binder_properties_%A.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

#!/bin/bash
# --- 04_extract_binder_properties.sh ---
# Amaç: PDBbind yapılarını tarayarak afinite (pK_d) ve dosya yollarını içeren JSON haritasını oluşturmak.
echo "Is basladi: $(date)"

source ~/.bashrc

conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# Script'in bulunduğu dizine git (önemli)
cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/04_extract_binder_properties.py

echo "Is bitti: $(date)"