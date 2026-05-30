#!/bin/bash
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A hnaydin
#SBATCH -J 06_pocket_banana
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pocket_banana_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pocket_banana_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G

echo "Is basladi: $(date)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/banana

export NUM_CPUS=${SLURM_CPUS_PER_TASK:-40}

python3 /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/06_pocket_create_and_score.py \
    -n $NUM_CPUS

if [ $? -ne 0 ]; then
    echo "HATA: Script basarisiz oldu."
    exit 1
fi

echo "Is bitti: $(date)"
