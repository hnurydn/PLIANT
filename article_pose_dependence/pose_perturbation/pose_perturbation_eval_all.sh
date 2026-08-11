#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=pose_perturb_eval
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/pose_perturb_eval_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/pose_perturb_eval_%j.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model

# 3 model x 5 RMSD bin = 15 degerlendirme, TEK checkpoint (iter-1900000), tek GPU job.
# baseline = negatif kontrol (lig_coordinates'i forward'da hic kullanmiyor -> RMSD'ye
# gore DUZ CIZGI vermesi beklenir, bkz. makale/pose_perturbation_teknik_tasarim.md §2).

echo "İş Başladı: $(date)  Host: $(hostname)"
eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# ─── Native valid.lmdb + perturbe edilmis core setlerini node-local SSD'ye kopyala ───
LMDB_NOESM_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet"
LMDB_ESM_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
PERTURB_NOESM_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_output"
PERTURB_ESM_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_output_esm"

SSD_DIR="/tmp/hnaydin_poseperturb_${SLURM_JOB_ID}"
mkdir -p $SSD_DIR/noesm_valid $SSD_DIR/esm_valid $SSD_DIR/noesm_test $SSD_DIR/esm_test

echo "--- Native valid.lmdb kopyalanıyor ---"
cp -r $LMDB_NOESM_DIR/valid.lmdb      $SSD_DIR/noesm_valid/
cp    $LMDB_NOESM_DIR/valid_meta.json $SSD_DIR/noesm_valid/
cp -r $LMDB_ESM_DIR/valid.lmdb        $SSD_DIR/esm_valid/
cp    $LMDB_ESM_DIR/valid_meta.json   $SSD_DIR/esm_valid/

echo "--- Perturbe edilmiş core setleri kopyalanıyor ---"
cp -r $PERTURB_NOESM_DIR/*.lmdb $PERTURB_NOESM_DIR/*.json $SSD_DIR/noesm_test/
cp -r $PERTURB_ESM_DIR/*.lmdb   $PERTURB_ESM_DIR/*.json   $SSD_DIR/esm_test/
echo "--- Kopyalama tamamlandı: $(date) ---"

SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/pose_perturbation_eval.py"
SCRIPTS_PATH="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts"
CKPT_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model"
OUT_CSV="/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_eval_results.csv"

# ─── 1. Baseline (NoESM_NoUW) — NEGATİF KONTROL: düz çizgi beklenir ───────────
echo "=== [1/3] Baseline (NoESM_NoUW) başlıyor: $(date) ==="
python $SCRIPT \
    --model_type noesm --model_label "NoESM_NoUW_baseline" \
    --ckpt_path  $CKPT_BASE/noESM_noUW_1:11_512/PLANET.iter-1900000 \
    --valid_data $SSD_DIR/noesm_valid/valid.lmdb \
    --test_data_dir $SSD_DIR/noesm_test \
    --rmsd_bins 0,1,2,3,4 --dims 512 \
    --scripts_path $SCRIPTS_PATH --out_csv $OUT_CSV
echo "=== [1/3] tamamlandı: $(date) ==="

# ─── 2. ESM_VG — poz-duyarlı olması beklenen ana model ────────────────────────
echo "=== [2/3] ESM_VG başlıyor: $(date) ==="
python $SCRIPT \
    --model_type vg --model_label "ESM_VG" \
    --ckpt_path  $CKPT_BASE/ESM2_UW_VECTOR_GATING_1:11/PLANET.iter-1900000 \
    --valid_data $SSD_DIR/esm_valid/valid.lmdb \
    --test_data_dir $SSD_DIR/esm_test \
    --rmsd_bins 0,1,2,3,4 --dims 512 \
    --scripts_path $SCRIPTS_PATH --out_csv $OUT_CSV
echo "=== [2/3] tamamlandı: $(date) ==="

# ─── 3. v4_Projection — kontrastif+projection varyantı, poz-duyarlı ───────────
echo "=== [3/3] v4_Projection başlıyor: $(date) ==="
python $SCRIPT \
    --model_type contrastive_v4 --model_label "v4_Projection" \
    --ckpt_path  $CKPT_BASE/ESM2_VG_CONTRASTIVE_v4_projection/PLANET.iter-1900000 \
    --valid_data $SSD_DIR/esm_valid/valid.lmdb \
    --test_data_dir $SSD_DIR/esm_test \
    --rmsd_bins 0,1,2,3,4 --dims 512 \
    --scripts_path $SCRIPTS_PATH --out_csv $OUT_CSV
echo "=== [3/3] tamamlandı: $(date) ==="

rm -rf $SSD_DIR
echo "İş Tamamlandı: $(date)"
echo "Sonuçlar: $OUT_CSV"
