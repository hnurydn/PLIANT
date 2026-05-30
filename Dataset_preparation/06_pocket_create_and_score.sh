#!/bin/bash
#SBATCH -p debug
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A hnaydin
#SBATCH -J 06_pocket_banana
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pocket_banana_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pocket_banana_%A.err
#SBATCH --time=2:00:00
#SBATCH --mem=50G

echo "Is basladi: $(date)"
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/banana

PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/06_pocket_create_and_score.py"

python3 "$PYTHON_SCRIPT" -n 20

echo "Is bitti: $(date)"
