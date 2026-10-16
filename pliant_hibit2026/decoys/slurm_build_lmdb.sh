#!/bin/bash
# Builds lmdb_planet_v2_cepte from the FULL set of placed decoy poses (core -> valid -> train).
# The old lmdb_planet_v2 is left untouched. The builder reads kuyruk_{split}_tam.txt as its queue.
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --job-name=lmdb_tam
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/logs/lmdb_tam_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/logs/lmdb_tam_%j.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte

echo "=== lmdb_planet_v2_cepte (TAM SET) : $(date) | $(hostname) ==="
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
export OMP_NUM_THREADS=1
D=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2_cepte
# remove the truncated (maks100) copies from 19 Sep - the full set is written in their place
rm -rf $D/core.lmdb $D/valid.lmdb $D/train.lmdb $D/core_meta.json $D/valid_meta.json $D/train_meta.json
for S in core valid train; do
  case $S in core) M=8 ;; valid) M=20 ;; train) M=90 ;; esac
  echo "--- $S (mapsize ${M}G) : $(date) ---"
  SPLIT=$S KIP=cepte MAPSIZE_GB=$M python -u build_inpocket_lmdb.py || exit 1
done
echo "--- boyutlar ---"; du -sh $D/*.lmdb
echo "=== Bitti: $(date) ==="
