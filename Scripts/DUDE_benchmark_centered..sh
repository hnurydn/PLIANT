#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=dude_centered_6model
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_centered_6model_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude_centered_6model_%j.err
#SBATCH -A hnaydin

echo "İş Başladı: $(date)"
eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model"
SCRIPT="$BASE/Scripts/DUDE_benchmark_centered.py"
OUT_DIR="$BASE/dude_results_centered"
mkdir -p $OUT_DIR

SSD="/tmp/hnaydin_dude_c_${SLURM_JOB_ID}"
SSD_NOESM="$SSD/noesm"
SSD_ESM="$SSD/esm"
mkdir -p $SSD_NOESM $SSD_ESM

echo "--- SSD kopyalama: $(date) ---"
cp -r $BASE/datasets/pdbbind2020/dude/lmdb_dude/*     $SSD_NOESM/
cp -r $BASE/datasets/pdbbind2020/dude/lmdb_dude_esm/* $SSD_ESM/
echo "--- Kopyalama bitti: $(date) ---"

CKPT_NOESM_NOUW="$BASE/noESM_noUW_1:11_512/PLANET.iter-1900000"
CKPT_NOESM_UW="$BASE/noESM_UW_1:11_512/PLANET.iter-1900000"
CKPT_ESM_NOUW="$BASE/esm_nouw/PLANET.iter-1900000"
CKPT_ESM_UW="$BASE/ESM2_UW_1:11/PLANET.iter-1900000"
CKPT_ESM_VG="$BASE/ESM2_UW_VECTOR_GATING_1:11/PLANET.iter-1900000"
CKPT_ESM_VG_CON="$BASE/ESM2_VG_CONTRASTIVE_1b11d/PLANET.iter-1900000"

# ─── Çift 3: ESM_VG + ESM_VG_Contrastive ─────────────────────────────────────
echo "=== ÇİFT 3 başlıyor: $(date) ==="
python $SCRIPT --model_type vg --lmdb_dir $SSD_ESM \
    --model_path $CKPT_ESM_VG --out_dir $OUT_DIR \
    --run_name ESM_VG_centered --center_ligands \
    --wandb_project PLANET_DUDE_CENTERED &
PID5=$!

python $SCRIPT --model_type contrastive --lmdb_dir $SSD_ESM \
    --model_path $CKPT_ESM_VG_CON --out_dir $OUT_DIR \
    --run_name ESM_VG_Contrastive_centered --center_ligands \
    --wandb_project PLANET_DUDE_CENTERED &
PID6=$!

wait $PID5 $PID6
echo "=== ÇİFT 3 bitti: $(date) ==="

# ─── Çift 1: noESM_noUW + noESM_UW ──────────────────────────────────────────
echo "=== ÇİFT 1 başlıyor: $(date) ==="
python $SCRIPT --model_type noesm --lmdb_dir $SSD_NOESM \
    --model_path $CKPT_NOESM_NOUW --out_dir $OUT_DIR \
    --run_name noESM_noUW_centered --center_ligands \
    --wandb_project PLANET_DUDE_CENTERED &
PID1=$!

python $SCRIPT --model_type noesm --lmdb_dir $SSD_NOESM \
    --model_path $CKPT_NOESM_UW --out_dir $OUT_DIR \
    --run_name noESM_UW_centered --center_ligands \
    --wandb_project PLANET_DUDE_CENTERED &
PID2=$!

wait $PID1 $PID2
echo "=== ÇİFT 1 bitti: $(date) ==="

# ─── Çift 2: ESM_noUW + ESM_UW ───────────────────────────────────────────────
echo "=== ÇİFT 2 başlıyor: $(date) ==="
python $SCRIPT --model_type esm --lmdb_dir $SSD_ESM \
    --model_path $CKPT_ESM_NOUW --out_dir $OUT_DIR \
    --run_name ESM_noUW_centered --center_ligands \
    --wandb_project PLANET_DUDE_CENTERED &
PID3=$!

python $SCRIPT --model_type esm --lmdb_dir $SSD_ESM \
    --model_path $CKPT_ESM_UW --out_dir $OUT_DIR \
    --run_name ESM_UW_centered --center_ligands \
    --wandb_project PLANET_DUDE_CENTERED &
PID4=$!

wait $PID3 $PID4
echo "=== ÇİFT 2 bitti: $(date) ==="



rm -rf $SSD
echo "İş Tamamlandı: $(date)"