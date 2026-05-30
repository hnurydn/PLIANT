#!/bin/bash
# --- 08_01_prepare_planet_data.sh ---
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20
#SBATCH --job-name=PLANET_Finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_Finalize_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_Finalize_%j.err
#SBATCH -A hnaydin

echo "Birinci islem baslatiliyor: $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 1. Komut: create_final_splits.py
python3 /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/create_final_splits.py \
    --input_json /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json \
    --core_list /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/casf2016_input_PLANET.csv \
    --output_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index

if [ $? -ne 0 ]; then
    echo "HATA: create_final_splits.py başarısız oldu. İş durduruluyor."
    exit 1
fi

echo "Ikinci islem baslatiliyor: $(date)"
# 2. Komut: PLANET_datautils.py
python3 /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/PLANET_datautils.py \
      --dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index \
      --pdbdir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020

echo "Tum islemler tamamlandi: $(date)"