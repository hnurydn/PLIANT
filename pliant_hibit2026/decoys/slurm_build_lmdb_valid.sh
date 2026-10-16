#!/bin/bash
# Builds the two in-pocket LMDBs for the valid_temiz split (22 Sep 2026). Runs only AFTER the
# placement job (6397208) finishes (--dependency).
# First checks that every shard in the queue has its npz. If any is missing it writes NOTHING
# (exit 6). It also refuses to write if the target valid_temiz.lmdb already exists.
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -A hnaydin
#SBATCH --job-name=lmdb_vt_cepte
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/logs/lmdb_vt_cepte_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/logs/lmdb_vt_cepte_%j.err
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte
set -u
P=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020
K=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/yerlesim
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
export OMP_NUM_THREADS=1 PYTHONPATH=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts
echo "=== $(date) $(hostname) ==="
eksik=0; n=0
while IFS=$'\t' read -r k a b; do [ -z "$k" ] && continue; n=$((n+1)); [ -f "$K/son/valid_temiz_${k}_${a}.npz" ] || eksik=$((eksik+1)); done < $K/kuyruk_valid_temiz.txt
echo "yerlesim: $n dilim, eksik npz $eksik"
[ $eksik -eq 0 ] || { echo "=== EKSIK DILIM VAR, LMDB YAZILMADI ==="; exit 6; }
for d in lmdb_planet_v2_cepte lmdb_planet_cepte; do [ -e $P/$d/valid_temiz.lmdb ] && { echo "HATA: $P/$d/valid_temiz.lmdb zaten var"; exit 2; }; done
export SPLIT=valid_temiz KIP=cepte KUYRUK_DOSYA=$K/kuyruk_valid_temiz.txt MAPSIZE_GB=20
SRC=$P/lmdb_planet_v2 DST=$P/lmdb_planet_v2_cepte python -u build_inpocket_lmdb.py || exit 1
SRC=$P/lmdb_planet    DST=$P/lmdb_planet_cepte    python -u build_inpocket_lmdb.py || exit 1
du -sh $P/lmdb_planet_v2_cepte/valid_temiz.lmdb $P/lmdb_planet_cepte/valid_temiz.lmdb
echo "--- dogrulama ---"
python -u -W ignore dogrula_vt_cepte.py
echo "--- egitimdeki veri denetimi (20 boyut) ---"
python -W ignore bl_veri_denetim.py $P/lmdb_planet_cepte/valid_temiz.lmdb 20 60; echo "rc=$?"
echo "=== Bitti $(date) ==="
