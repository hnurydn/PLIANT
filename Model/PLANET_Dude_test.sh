#!/bin/bash
# --- PLANET_dude_test.sh ---
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=48  # dude.py script'i 48 çekirdekle çalışacak (njobs=48)
#SBATCH --job-name=DUDE_VS_TEST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_test_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_test_%j.err
#SBATCH -A hnaydin

echo "Is basladi (DUDE VS TEST): $(date)"
echo "Host: $(hostname)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# --- PATH TANIMLARI ---
# DUD-E'nin alt klasörlerini içeren ana dizin
DUDE_DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/all"

# Eğitilmiş modelinizin (checkpoint) tam yolu
PLANET_CHECKPOINT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL/PLANET.iter-500000"

# Sonuçların kaydedileceği dizin
RESULTS_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/DUDE_SCREENING_RESULTS"

# dude.py script'inizin yolu
DUDE_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE.py"

mkdir -p $RESULTS_DIR

# --- PYTHON YOLUNU AYARLAMA ---
# dude.py script'i PLANET_model'ı bulmak için Model klasörünü path'e eklemelidir.
# Bu düzeltmeyi doğrudan komut satırında yapıyoruz.
PYTHON_SCRIPT="import sys; sys.path.append('/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation'); import dude as dude_script; dude_script.main()"


python $DUDE_SCRIPT \
-f $PLANET_CHECKPOINT \
-d $DUDE_DATA_ROOT \
-o $RESULTS_DIR \
-s 300 -n 8 -k 300 -va 300 -pu 3 -lu 10 -pl 1

echo "Is bitti (DUDE VS TEST): $(date)"