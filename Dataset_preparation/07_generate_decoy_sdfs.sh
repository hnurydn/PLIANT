#!/bin/bash
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A hnaydin
#SBATCH -J 07_generate_decoy_sdfs
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_decoy_sdfs_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_decoy_sdfs_%A.err
#SBATCH --time=02:00:00 
#SBATCH --mem=24G

# TEST ETMEK İSTEDİĞİNİZ PDB ID'sini BURAYA YAZIN
# TEST_PDB="2qbr" 
# PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/07_generate_decoy_sdfs_deneme.py"
# python3 "$PYTHON_SCRIPT" -n $NUM_CPUS -p "$TEST_PDB"


echo "Is basladi: $(date)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

export NUM_CPUS=8 
PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/07_generate_decoy_sdfs_deneme.py"

# -p argümanı ile PDB ID'sini Python scriptine gönderiyoruz
python3 "$PYTHON_SCRIPT" -n $NUM_CPUS

echo "Is bitti: $(date)"