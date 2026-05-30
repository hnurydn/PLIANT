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

# 1. Değişkenler (LIT-PCBA)
# Kaynak Klasör (Arf üzerindeki yavaş disk)
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude"

# Hedef Klasörler (SSD ve Sonuç Çıktısı)
LOCAL_SSD="/tmp/${USER}_dude_eval"
OUT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/dude/benchmark_results"

# Dosya Yolları
ORIG_JSON="$SCRATCH_BASE/dude_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/dude_paths_ssd.json"

# 🎯 H5 Dosyası Ayarı:
# Python script'e vereceğimiz yol, SSD üzerindeki yol olmalı!
# Dosya rsync ile taşınacağı için doğrudan SSD yolunu tanımlıyoruz.
H5_PATH="$LOCAL_SSD/dude_esm2_embeddings.h5"

echo "📂 Klasörler oluşturuluyor..."
mkdir -p $LOCAL_SSD
mkdir -p $OUT_DIR

echo "📂 Veriler SSD'ye kopyalanıyor (Bu işlem veri boyutuna göre sürebilir)..."
# LIT-PCBA klasörünün İÇİNDEKİLERİ SSD'ye atıyoruz.
# H5 dosyası da bu klasörün içindeyse o da kopyalanır.
rsync -aq "$SCRATCH_BASE/" "$LOCAL_SSD/"

# 2. JSON Yollarını SSD'ye Göre Güncelle
# JSON içindeki /arf/... yolunu /tmp/... ile değiştirip yeni bir JSON oluşturuyoruz.
sed "s|$SCRATCH_BASE|$LOCAL_SSD|g" "$ORIG_JSON" > "$TEMP_JSON"

echo "✅ Veri transferi bitti."
echo "✅ SSD JSON Yolu: $TEMP_JSON"
echo "✅ SSD H5 Yolu: $H5_PATH"

source ~/.bashrc
conda activate pocketgen

# 🎯 Python Script'inin sunucudaki tam yolu (LIT-PCBA ESM versiyonu)
PY_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE_benchmark_DEBUG.py"

# 1. Model 1: ESM_UW Çalıştırma
echo "🚀 Model 1: ESM_UW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_ESM2_UW/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name dude_ESM_UW \
--h5_path $H5_PATH \
--batch_size 32 \
--feature_dims 512 \
--device cpu

# 2. Model 2: ESM_NoUW Çalıştırma
echo "🚀 Model 2: ESM_NoUW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_ESM2_NoUW/PLANET_ESM2_Original.iter-1190000 \
--out_dir $OUT_DIR \
--out_name dude_ESM_NoUW \
--h5_path $H5_PATH \
--batch_size 32 \
--feature_dims 512 \
--device cpu

# 3. Model 3: ESM_UW_VECTOR_Dropout_GATING Çalıştırma
echo "🚀 Model 3: ESM_UW_VECTOR_Dropout_GATING Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name dude_ESM_UW_VECTOR_Dropout_GATING \
--h5_path $H5_PATH \
--batch_size 32 \
--feature_dims 512 \
--device cpu

# 4. Model 4: ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS Çalıştırma
echo "🚀 Model 4: ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2_VECTOR_GATING_CONTRASTIVE_LOSS/PLANET_ESM2.iter-1199400 \
--out_dir $OUT_DIR \
--out_name dude_ESM_UW_VECTOR_Dropout_GATING_CONTRASTIVE_LOSS\
--h5_path $H5_PATH \
--batch_size 32 \
--feature_dims 512 \
--device cpu

# 🎯 ÖNEMLİ: Python script isminin sunucudakiyle aynı olduğundan emin ol!
PY_SCRIPT_2="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE_benchmark_No_ESM.py"

# 5. Model 5: No_ESM_UW (Yol ve İsim Eşleşti)
echo "🚀 Model 5: No_ESM_UW Başlatılıyor..."
python $PY_SCRIPT_2 \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_UW/PLANET.iter-1240000 \
--out_dir $OUT_DIR \
--out_name dude_NoESM_UW \
--feature_dims 300

# 6. Model 6: No_ESM_NoUW (Yol ve İsim Eşleşti)
echo "🚀 Model 6: No_ESM_NoUW Başlatılıyor..."
python $PY_SCRIPT_2 \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_NoUW/PLANET.iter-1400000 \
--out_dir $OUT_DIR \
--out_name dude_NoESM_NoUW \
--feature_dims 300

# 5. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"