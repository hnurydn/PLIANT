#!/bin/bash
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54 
#SBATCH --job-name=planet_litpcba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_litpcba_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_litpcba_%j.err
#SBATCH -A hnaydin

# 1. Değişkenler
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA"
OUT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/litpcba/benchmark_results"
LOCAL_SSD="/tmp/${USER}_litpcba_eval"

ORIG_JSON="$SCRATCH_BASE/lit_pcba_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/lit_pcba_paths_ssd.json"
H5_PATH="$LOCAL_SSD/lit_pcba_esm2_embeddings.h5"

echo "📂 Creating folders..."
mkdir -p $LOCAL_SSD
mkdir -p $OUT_DIR

echo "📂 Copying data to SSD..."
rsync -aq "$SCRATCH_BASE/" "$LOCAL_SSD/"

sed "s|$SCRATCH_BASE|$LOCAL_SSD|g" "$ORIG_JSON" > "$TEMP_JSON"

echo "✅ Data transfer complete."
source ~/.bashrc
conda activate pocketgen

PY_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/LITPCBA_benchmark_ESM.py"

# --- MODELLER (PARAMETRELER TEMİZLENDİ) ---

# 3. Model 3: GATING
echo "🚀 Model 3: ESM_UW_VECTOR_Dropout_GATING Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_UW_VECTOR_Dropout_GATING \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 24 \
--feature_dims 512

# 4. Model 4: CONTRASTIVE LOSS
echo "🚀 Model 4: ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING_CONTRASTIVE_LOSS/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 24 \
--feature_dims 512

# 5. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"