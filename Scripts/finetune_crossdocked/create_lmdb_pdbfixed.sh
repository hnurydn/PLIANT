#!/bin/bash
#SBATCH --job-name=create_lmdb_pdbfixed
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/create_lmdb_pdbfixed_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/create_lmdb_pdbfixed_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen


BASE=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model
SCRIPTS=$BASE/dude_data_preparation
H5=$BASE/pdbbind2020_esm2_embeddings_v2.h5
PCA=$BASE/esm2_pca_params.pkl

# ─── DUDE NoESM ───────────────────────────────────────────────────────────────
echo "=== DUDE NoESM LMDB oluşturuluyor ==="
python $SCRIPTS/create_dude_lmdb.py \
    --json_path $BASE/datasets/pdbbind2020/dude/dude_paths_pdbfixed_final.json \
    --out_lmdb  $BASE/datasets/pdbbind2020/dude/lmdb_dude_pdbfixed \
    --model_type noesm \
    --cutoff 12.0
echo "=== DUDE NoESM tamamlandı: $(date) ==="

# ─── DUDE ESM ─────────────────────────────────────────────────────────────────
echo "=== DUDE ESM LMDB oluşturuluyor ==="
python $SCRIPTS/create_dude_lmdb_esm.py \
    --json_path $BASE/datasets/pdbbind2020/dude/dude_paths_pdbfixed_final.json \
    --h5_path   $H5 \
    --pca_path  $PCA \
    --out_lmdb  $BASE/datasets/pdbbind2020/dude/lmdb_dude_esm_pdbfixed
echo "=== DUDE ESM tamamlandı: $(date) ==="

# ─── LIT-PCBA NoESM ───────────────────────────────────────────────────────────
echo "=== LIT-PCBA NoESM LMDB oluşturuluyor ==="
python $SCRIPTS/create_litpcba_lmdb.py \
    --json_path  $BASE/datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_pdbfixed_final.json \
    --out_lmdb   $BASE/datasets/pdbbind2020/LIT-PCBA/lmdb_litpcba_noesm_pdbfixed \
    --model_type noesm
echo "=== LIT-PCBA NoESM tamamlandı: $(date) ==="

# ─── LIT-PCBA ESM ─────────────────────────────────────────────────────────────
echo "=== LIT-PCBA ESM LMDB oluşturuluyor ==="
python $SCRIPTS/create_litpcba_lmdb.py \
    --json_path  $BASE/datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_pdbfixed_final.json \
    --h5_path    $H5 \
    --pca_path   $PCA \
    --out_lmdb   $BASE/datasets/pdbbind2020/LIT-PCBA/lmdb_litpcba_esm_pdbfixed \
    --model_type esm
echo "=== LIT-PCBA ESM tamamlandı: $(date) ==="

echo "İş Tamamlandı: $(date)"