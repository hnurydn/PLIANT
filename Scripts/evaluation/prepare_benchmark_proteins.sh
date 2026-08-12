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


BASE=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model
SCRIPT=$BASE/Scripts/prepare_benchmark_proteins.py

# ─── DUDE ─────────────────────────────────────────────────────────────────────
echo "=== DUDE proteinleri hazırlanıyor ==="
python $SCRIPT \
    --benchmark dude \
    --json_path $BASE/datasets/pdbbind2020/dude/dude_paths_prepared_arf.json \
    --njobs 20
echo "=== DUDE tamamlandı: $(date) ==="

# ─── LIT-PCBA ─────────────────────────────────────────────────────────────────
echo "=== LIT-PCBA proteinleri hazırlanıyor ==="
python $SCRIPT \
    --benchmark litpcba \
    --json_path $BASE/datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_arf.json \
    --njobs 20
echo "=== LIT-PCBA tamamlandı: $(date) ==="

echo "İş Tamamlandı: $(date)"