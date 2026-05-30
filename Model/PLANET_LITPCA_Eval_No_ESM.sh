#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20 
#SBATCH --job-name=planet_litpcba_no_esm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_litpcba_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_litpcba_%j.err
#SBATCH -A hnaydin

# 1. Değişkenler
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA"
OUT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA/benchmark_results"
LOCAL_SSD="/tmp/${USER}_litpcba_eval"
ORIG_JSON="$SCRATCH_BASE/lit_pcba_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/lit_pcba_paths_ssd.json"

echo "📂 Veriler SSD'ye kopyalanıyor..."
mkdir -p $LOCAL_SSD
mkdir -p $OUT_DIR
# 🎯 DÜZELTME: $SCRATCH_BASE kullanıyoruz
rsync -aq $SCRATCH_BASE/ $LOCAL_SSD/

# rsync satırından hemen sonra ekle
echo "🔍 SSD Kontrolü (VDR Klasörü):"
ls -lh $LOCAL_SSD/VDR | head -n 5

# 2. JSON Yollarını SSD'ye Göre Güncelle
sed "s|/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA|$LOCAL_SSD|g" $ORIG_JSON > $TEMP_JSON

echo "✅ JSON yolları SSD'ye uyarlandı: $TEMP_JSON"

source ~/.bashrc
conda activate pocketgen 

# 🎯 ÖNEMLİ: Python script isminin sunucudakiyle aynı olduğundan emin ol!
PY_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/LIT-PCBA_benchmark_No_ESM.py"

# 🎯 Model 1: No_ESM_UW (Yol ve İsim Eşleşti)
echo "🚀 Model 1: No_ESM_UW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_UW/PLANET.iter-1940000 \
--out_dir $OUT_DIR \
--out_name lit_pcba_NoESM_UW \
--json_path $TEMP_JSON \
--feature_dims 300

# 🎯 Model 2: No_ESM_NoUW (Yol ve İsim Eşleşti)
echo "🚀 Model 2: No_ESM_NoUW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_NoUW/PLANET.iter-1940000 \
--out_dir $OUT_DIR \
--out_name lit_pcba_NoESM_NoUW \
--json_path $TEMP_JSON \
--feature_dims 300

# 5. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"