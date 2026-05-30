#!/bin/bash
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J 20_prepare_lit_pcba
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/20_prepare_lit_pcba_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/20_prepare_lit_pcba_%A.err
#SBATCH --time=1-00:00:00 
#SBATCH --mem=100G

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

echo "LIT-PCBA PKL Olusturma Basliyor..."
python -u /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/20_prepare_lit_pcba_pkl.py
