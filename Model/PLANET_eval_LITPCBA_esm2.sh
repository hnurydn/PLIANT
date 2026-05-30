#!/bin/bash
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20 
#SBATCH --job-name=planet_litpcba_esm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_litpcba_esm_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_litpcba_esm_%j.err
#SBATCH -A hnaydin

# 1. Değişkenler (LIT-PCBA Olarak Güncellendi)
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA"
OUT_DIR="$SCRATCH_BASE/benchmark_results"
LOCAL_SSD="/tmp/${USER}_litpcba_eval"
ORIG_JSON="$SCRATCH_BASE/lit_pcba_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/lit_pcba_paths_ssd.json"
# 🎯 Onardığın yeni H5 dosyasının yolu:
H5_PATH="$SCRATCH_BASE/lit_pcba_esm2_embeddings.h5"

echo "📂 Veriler SSD'ye kopyalanıyor..."
mkdir -p $LOCAL_SSD
mkdir -p $OUT_DIR
# LIT-PCBA klasörünü SSD'ye taşı (Okuma hızını uçurur!)
rsync -aq $SCRATCH_BASE/ $LOCAL_SSD/

# 2. JSON Yollarını SSD'ye Göre Güncelle (LIT-PCBA path'ini hedef alıyoruz)
sed "s|/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA|$LOCAL_SSD|g" $ORIG_JSON > $TEMP_JSON

echo "✅ JSON yolları SSD'ye uyarlandı: $TEMP_JSON"

source ~/.bashrc
conda activate pocketgen

# 🎯 Python Script'inin sunucudaki tam yolu (LIT-PCBA ESM versiyonu)
PY_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/LITPCBA_benchmark_ESM.py"

# 3. Model 1: ESM_UW Çalıştırma
echo "🚀 Model 1: ESM_UW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_ESM2_UW/PLANET_ESM2.iter-1939030 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_UW \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 128 \
--feature_dims 512

# 4. Model 2: ESM_NoUW Çalıştırma
echo "🚀 Model 2: ESM_NoUW Başlatılıyor..."
python $PY_SCRIPT \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_ESM2_NoUW/PLANET_ESM2_Original.iter-1940000 \
--out_dir $OUT_DIR \
--out_name lit_pcba_ESM_NoUW \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--batch_size 128 \
--feature_dims 512

# 5. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"