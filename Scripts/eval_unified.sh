#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=planet_threshold_eval
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/eval_unified_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/eval_unified_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
echo "Host: $(hostname)"

eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# ─── Veriyi /tmp SSD'ye kopyala ───────────────────────────────────────────────
LMDB_NOESM="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet"
LMDB_ESM="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"

SSD_DIR="/tmp/hnaydin_thresh_${SLURM_JOB_ID}"
SSD_NOESM=$SSD_DIR/noesm
SSD_ESM=$SSD_DIR/esm
mkdir -p $SSD_NOESM $SSD_ESM

echo "--- NoESM LMDB SSD'ye kopyalanıyor ---"
cp -r $LMDB_NOESM/valid.lmdb      $SSD_NOESM/
cp    $LMDB_NOESM/valid_meta.json  $SSD_NOESM/
cp -r $LMDB_NOESM/core.lmdb       $SSD_NOESM/
cp    $LMDB_NOESM/core_meta.json   $SSD_NOESM/

echo "--- ESM LMDB SSD'ye kopyalanıyor ---"
cp -r $LMDB_ESM/valid.lmdb        $SSD_ESM/
cp    $LMDB_ESM/valid_meta.json    $SSD_ESM/
cp -r $LMDB_ESM/core.lmdb         $SSD_ESM/
cp    $LMDB_ESM/core_meta.json     $SSD_ESM/
echo "--- Kopyalama tamamlandı ---"

# Kısayollar
VALID_NOESM=$SSD_NOESM/valid.lmdb
TEST_NOESM=$SSD_NOESM/core.lmdb
VALID_ESM=$SSD_ESM/valid.lmdb
TEST_ESM=$SSD_ESM/core.lmdb

SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts/eval_unified.py"
CKPT_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model"

# ─── 1. NoESM_NoUW ────────────────────────────────────────────────────────────
echo "=== [1/6] NoESM_NoUW başlıyor: $(date) ==="
python $SCRIPT \
    --model_type noesm \
    --ckpt_dir   $CKPT_BASE/noESM_noUW_1:11_512 \
    --valid_data $VALID_NOESM \
    --test_data  $TEST_NOESM \
    --run_name   noESM_noUW_threshold \
    --dims       512 \
    --min_step   1360000
echo "=== [1/6] NoESM_NoUW tamamlandı: $(date) ==="

# ─── 2. NoESM_UW ──────────────────────────────────────────────────────────────
echo "=== [2/6] NoESM_UW başlıyor: $(date) ==="
python $SCRIPT \
    --model_type noesm \
    --ckpt_dir   $CKPT_BASE/noESM_UW_1:11_512 \
    --valid_data $VALID_NOESM \
    --test_data  $TEST_NOESM \
    --run_name   noESM_UW_threshold \
    --dims       512 \
    --min_step   1360000
echo "=== [2/6] NoESM_UW tamamlandı: $(date) ==="

# ─── 3. ESM_NoUW ──────────────────────────────────────────────────────────────
echo "=== [3/6] ESM_NoUW başlıyor: $(date) ==="
python $SCRIPT \
    --model_type esm \
    --ckpt_dir   $CKPT_BASE/esm_nouw \
    --valid_data $VALID_ESM \
    --test_data  $TEST_ESM \
    --run_name   ESM_noUW_threshold \
    --dims       512 \
    --min_step   1360000
echo "=== [3/6] ESM_NoUW tamamlandı: $(date) ==="

# ─── 4. ESM_UW ────────────────────────────────────────────────────────────────
echo "=== [4/6] ESM_UW başlıyor: $(date) ==="
python $SCRIPT \
    --model_type esm \
    --ckpt_dir   $CKPT_BASE/ESM2_UW_1:11 \
    --valid_data $VALID_ESM \
    --test_data  $TEST_ESM \
    --run_name   ESM_UW_threshold \
    --dims       512 \
    --min_step   1360000
echo "=== [4/6] ESM_UW tamamlandı: $(date) ==="

# ─── 5. ESM_VG ────────────────────────────────────────────────────────────────
echo "=== [5/6] ESM_VG başlıyor: $(date) ==="
python $SCRIPT \
    --model_type vg \
    --ckpt_dir   $CKPT_BASE/ESM2_UW_VECTOR_GATING_1:11 \
    --valid_data $VALID_ESM \
    --test_data  $TEST_ESM \
    --run_name   ESM_VG_threshold \
    --dims       512 \
    --min_step   1360000
echo "=== [5/6] ESM_VG tamamlandı: $(date) ==="

# ─── 6. ESM_VG_Contrastive ────────────────────────────────────────────────────
echo "=== [6/6] ESM_VG_Contrastive başlıyor: $(date) ==="
python $SCRIPT \
    --model_type contrastive \
    --ckpt_dir   $CKPT_BASE/ESM2_VG_CONTRASTIVE_1b11d \
    --valid_data $VALID_ESM \
    --test_data  $TEST_ESM \
    --run_name   ESM_VG_Contrastive_threshold \
    --dims       512 \
    --min_step   1360000
echo "=== [6/6] ESM_VG_Contrastive tamamlandı: $(date) ==="

rm -rf $SSD_DIR
echo "İş Tamamlandı: $(date)"