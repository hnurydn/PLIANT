#!/bin/bash
#SBATCH --job-name=prepare_benchmark
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/prepare_benchmark_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/prepare_benchmark_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen


python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/prepare_crossdocked_pkls.py \
    --json_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_new_records.json \
    --extracted_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_extracted \
    --output_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_pkls \
    --njobs 40