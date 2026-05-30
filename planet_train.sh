#!/bin/bash
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=planet_train
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/train_%j.err
#SBATCH -A hnaydin

echo "Veri SSD ye kopyalanıyor..."
cp -r /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/train_tensor.lmdb /tmp/
cp -r /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/valid_tensor.lmdb /tmp/
cp -r /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/core_tensor.lmdb /tmp/
cp /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/train_tensor_meta.json /tmp/
cp /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/valid_tensor_meta.json /tmp/
cp /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/core_tensor_meta.json /tmp/
echo "Kopyalama tamamlandi."

SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/checkpoints"
mkdir -p $SAVE_DIR

/arf/home/hnaydin/miniconda3/envs/pocketgen/bin/python \
    /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_train_esm2_vector_gating_contrastive_loss.py \
    -t  /tmp/train_tensor.lmdb \
    -v  /tmp/valid_tensor.lmdb \
    -te /tmp/core_tensor.lmdb \
    -d  $SAVE_DIR \
    --batch_size 48 \
    --binder_per_batch 4 \
    --decoy_count 11 \
    --epoch 600

echo "Is bitti: $(date)"

# SSD temizle
echo "SSD temizleniyor..."
rm -rf /tmp/train_tensor.lmdb /tmp/valid_tensor.lmdb /tmp/core_tensor.lmdb
rm -f /tmp/train_tensor_meta.json /tmp/valid_tensor_meta.json /tmp/core_tensor_meta.json
echo "Temizlik tamamlandi."
