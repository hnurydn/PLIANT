#!/bin/bash
# --- PLANET_dude_gpu.sh ---
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --job-name=planet_dude_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G             # Memory
#SBATCH --time=1-00:00:00     # Time limit
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_dude_gpu_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_dude_gpu_%j.err
#SBATCH -A hnaydin

echo "Job Started (GPU DUDE TEST): $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# --- PATHS ---
# Path to your trained model checkpoint (non-ESM2)
MODEL_CHECKPOINT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL/PLANET.iter-50000" 
# Path to the DUDE dataset folder containing targets (e.g., ace, adr2b...)
DUDE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude"
# Output directory for results
OUT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_results_gpu"
mkdir -p $OUT_DIR

# --- RUN PYTHON SCRIPT ---
# Assuming you save the python code below as 'dude_gpu.py'
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/dude_gpu.py \
    -f "$MODEL_CHECKPOINT" \
    -d "$DUDE_DIR" \
    -o "$OUT_DIR" \
    --device cpu \
    --batch_size 32

echo "Job Finished: $(date)"