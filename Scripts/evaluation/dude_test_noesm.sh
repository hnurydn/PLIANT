#!/bin/bash
#SBATCH --job-name=dude_test_noesm
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/logs/dude_test_noesm_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/logs/dude_test_noesm_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

mkdir -p /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/logs
mkdir -p /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_results_test

CKPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/noESM_noUW_1:11_512/PLANET.iter-2000000"

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_data_preparation/DUDE_benchmark_NoESM_lmdb.py \
    --lmdb_dir   /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/lmdb_dude \
    --model_path $CKPT \
    --out_dir    /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_results_test \
    --feature_dims 512 \
    --nheads 8 \
    --key_dims 128 \
    --value_dims 128 \
    --pro_update_inters 3 \
    --lig_update_iters 10 \
    --pro_lig_update_iters 1 \
    --batch_size 64 \
    --device cpu \
    --wandb_project PLANET_DUDE_TEST

echo "İş Tamamlandı: $(date)"
