#!/bin/bash
# Amaç: Sadece tek bir PDB ID'si üzerinde Cep Oluşturma ve BANANA skorlama testini çalıştırmak.

#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH -A hnaydin
#SBATCH -J 07_generate_decoy_sdfs
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/rclone_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/rclone_%A.err
#SBATCH --time=1-00:00:00 
#SBATCH --mem=128G

# TEST ETMEK İSTEDİĞİNİZ PDB ID'sini BURAYA YAZIN
# TEST_PDB="2qbr" 
# PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/07_generate_decoy_sdfs_deneme.py"
# python3 "$PYTHON_SCRIPT" -n $NUM_CPUS -p "$TEST_PDB"


echo "Is basladi: $(date)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
# /usr/bin/rclone copy /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/data data_transfer:/Data/all_data -P
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/17_convert_decoys_inplace.py
