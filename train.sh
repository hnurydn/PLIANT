#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=PLANET_train_esm2_vector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_vector_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_train_esm2_vector_%j.err
#SBATCH -A hnaydin


source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/PLANET_train_esm2_vector_gating.py \
    -t /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/train_tensor.lmdb \
    -v /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/valid_tensor.lmdb \
    -te /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/core_tensor.lmdb \
    -d /tmp/test_output \
    --batch_size 24 --binder_per_batch 2 --decoy_count 11 \
    --epoch 1 --save_iter 10000 \
    --load_checkpoint "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING_1:11/PLANET_ESM2.iter-5027485" \
    --initial_step 5027485 --lr 0.00000059