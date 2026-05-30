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

# 1. Değişkenler (LIT-PCBA)
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA"
OUT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/litpcba/benchmark_results"
LOCAL_SSD="/tmp/${USER}_litpcba_eval"

# JSON ve H5 Yolları
ORIG_JSON="$SCRATCH_BASE/lit_pcba_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/lit_pcba_paths_ssd.json"
# SSD üzerindeki H5 yolu
H5_PATH="$LOCAL_SSD/lit_pcba_esm2_embeddings.h5"

echo "📂 Klasörler oluşturuluyor..."
mkdir -p $LOCAL_SSD
mkdir -p $OUT_DIR

echo "📂 Veriler SSD'ye kopyalanıyor..."
rsync -aq "$SCRATCH_BASE/" "$LOCAL_SSD/"

# JSON içindeki yolları güncelle
sed "s|$SCRATCH_BASE|$LOCAL_SSD|g" "$ORIG_JSON" > "$TEMP_JSON"

echo "✅ Veri transferi bitti. SSD H5 Yolu: $H5_PATH"

source ~/.bashrc
conda activate pocketgen

# Python Script Yolları
PY_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/LITPCBA_benchmark_ESM.py"
PY_SCRIPT_2="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/LIT-PCBA_benchmark_No_ESM.py"

# 1. Model 1: ESM_UW Çalıştırma
echo "🚀 Model 1: ESM_UW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_ESM2_UW/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_UW \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 24 \
--feature_dims 512

# 2. Model 2: ESM_NoUW Çalıştırma
echo "🚀 Model 2: ESM_NoUW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_ESM2_NoUW/PLANET_ESM2_Original.iter-1190000 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_NoUW \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 24 \
--feature_dims 512

# 3. Model 3: ESM_UW_VECTOR_Dropout_GATING Çalıştırma
echo "🚀 Model 3: ESM_UW_VECTOR_Dropout_GATING Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_UW_VECTOR_Dropout_GATING \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 24 \
--feature_dims 512

# 4. Model 4: ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS Çalıştırma
echo "🚀 Model 4: ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING_CONTRASTIVE_LOSS/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 24 \
--feature_dims 512

# 5. Model 5: No_ESM_UW (Yol ve İsim Eşleşti)
echo "🚀 Model 5: No_ESM_UW Başlatılıyor..."
python $PY_SCRIPT_2 \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_UW/PLANET.iter-1240000 \
--out_dir $OUT_DIR \
--out_name lit_pcba_NoESM_UW \
--json_path $TEMP_JSON \
--feature_dims 300

# 6. Model 6: No_ESM_NoUW (Yol ve İsim Eşleşti)
echo "🚀 Model 6: No_ESM_NoUW Başlatılıyor..."
python $PY_SCRIPT_2 \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_NoUW/PLANET.iter-1400000 \
--out_dir $OUT_DIR \
--out_name lit_pcba_NoESM_NoUW \
--json_path $TEMP_JSON \
--feature_dims 300

# 5. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"