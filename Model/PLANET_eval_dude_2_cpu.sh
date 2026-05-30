#!/bin/bash
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54 
#SBATCH --job-name=planet_dude
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_dude_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_dude_%j.err
#SBATCH -A hnaydin

# 1. Variables
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude"
OUT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude/benchmark_results"
LOCAL_SSD="/tmp/${USER}_dude_eval"
ORIG_JSON="$SCRATCH_BASE/dude_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/dude_paths_ssd.json"
H5_PATH="$LOCAL_SSD/dude_esm2_embeddings.h5"

echo "📂 Creating folders..."
mkdir -p $LOCAL_SSD
mkdir -p $OUT_DIR

echo "📂 Copying data to SSD..."
rsync -aq "$SCRATCH_BASE/" "$LOCAL_SSD/"

sed "s|$SCRATCH_BASE|$LOCAL_SSD|g" "$ORIG_JSON" > "$TEMP_JSON"

echo "✅ Data transfer complete."
source ~/.bashrc
conda activate pocketgen

PY_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE_benchmark_ESM.py"

# --- Model 3: GATING ---
echo "🚀 Model 3: ESM_UW_VECTOR_Dropout_GATING Starting..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name dude_ESM_UW_VECTOR_Dropout_GATING \
--h5_path $H5_PATH \
--batch_size 32 \
--feature_dims 512 \
--json_path $TEMP_JSON

# --- Model 4: CONTRASTIVE ---
echo "🚀 Model 4: CONTRASTIVE LOSS Starting..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING_CONTRASTIVE_LOSS/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name dude_ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS \
--h5_path $H5_PATH \
--batch_size 32 \
--feature_dims 512 \
--json_path $TEMP_JSON

# 5. Cleanup
rm -rf $LOCAL_SSD
echo "🧹 SSD Cleaned, Job Done!"