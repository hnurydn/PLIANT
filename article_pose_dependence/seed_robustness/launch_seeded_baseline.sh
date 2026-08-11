#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=seed_baseline
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/seed_baseline_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/seed_baseline_%j.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model

# 3-seed robustluk deneyi: NoESM_NoUW (baseline) modelini SEED (env var, --export ile
# gelir) ile TARGET_STEP'e kadar egitir. "Resume-aware": SAVE_DIR'de zaten checkpoint
# varsa oradan devam eder; TARGET_STEP'e zaten ulasilmissa hicbir sey yapmadan biter.
# Zincirdeki bir sonraki job (--dependency=afterany) walltime/crash farketmeksizin
# calisir ve ayni mantikla kaldigi yerden devam eder.

echo "İş Başladı: $(date)  Host: $(hostname)  SEED=${SEED}  TARGET_STEP=${TARGET_STEP}"
eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

DATA_ROOT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet"
SAVE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/noESM_noUW_1:11_512_seed${SEED}"
SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/PLANET_train_noESM_noUW_seeded.py"
SSD_DIR="/tmp/hnaydin_seed_baseline_${SEED}_${SLURM_JOB_ID}"

mkdir -p "$SAVE_DIR" "$SSD_DIR"

# ── Zaten hedefe ulasilmis mi kontrol et ──
LATEST_FILE=$(ls "$SAVE_DIR" 2>/dev/null | grep -E 'iter-[0-9]+$' | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_FILE" ]; then
    LATEST_STEP=$(echo "$LATEST_FILE" | grep -oE '[0-9]+$')
    echo "Mevcut en son checkpoint: $LATEST_FILE (step=$LATEST_STEP)"
    if [ "$LATEST_STEP" -ge "$TARGET_STEP" ]; then
        echo "🏁 Hedefe ($TARGET_STEP) zaten ulaşılmış, iş atlanıyor."
        rm -rf "$SSD_DIR"
        exit 0
    fi
    RESUME_ARGS="--load_checkpoint $SAVE_DIR/$LATEST_FILE --initial_step $LATEST_STEP"
else
    echo "Checkpoint yok, sıfırdan başlanıyor."
    RESUME_ARGS=""
fi

echo "--- LMDB SSD'ye kopyalanıyor ---"
cp -r "$DATA_ROOT/train.lmdb" "$SSD_DIR/"
cp    "$DATA_ROOT/train_meta.json" "$SSD_DIR/"
cp -r "$DATA_ROOT/valid.lmdb" "$SSD_DIR/"
cp    "$DATA_ROOT/valid_meta.json" "$SSD_DIR/"
cp -r "$DATA_ROOT/core.lmdb" "$SSD_DIR/"
cp    "$DATA_ROOT/core_meta.json" "$SSD_DIR/"
echo "--- Kopyalama bitti: $(date) ---"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

echo "--- Eğitim başlıyor: $(date) ---"
python "$SCRIPT" \
    -t "$SSD_DIR/train.lmdb" -v "$SSD_DIR/valid.lmdb" -te "$SSD_DIR/core.lmdb" \
    -d "$SAVE_DIR" \
    --decoy_count 11 --epoch 999 --anneal_iter 20000 --save_iter 20000 --print_iter 1000 \
    --seed "$SEED" --max_step "$TARGET_STEP" \
    $RESUME_ARGS

rm -rf "$SSD_DIR"
echo "İş Bitti: $(date)"
