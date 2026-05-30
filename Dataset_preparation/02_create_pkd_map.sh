#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH -A hnaydin
#SBATCH -J 02_create_pkd_map
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/create_pkd_map_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/create_pkd_map_%A.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

#!/bin/bash
# --- 02_create_pkd_map.sh ---
# Amaç: PDBbind yapılarını tarayarak afinite (pK_d) ve dosya yollarını içeren JSON haritasını oluşturmak.
echo "Is basladi: $(date)"

source ~/.bashrc

conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# Script'in bulunduğu dizine git (önemli)
cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/02_create_pkd_map.py

echo "Is bitti: $(date)"