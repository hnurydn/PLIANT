#!/bin/bash
# --- PLANET_eval_ESM2.sh ---
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --job-name=planet_esm2_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_esm2_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_eval_esm2_%j.err
#SBATCH -A hnaydin

# 1. Değişkenler
SCRATCH_BASE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA"
LOCAL_SSD="/tmp/${USER}_litpcba_eval"
ORIG_JSON="$SCRATCH_BASE/lit_pcba_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/lit_pcba_paths_ssd.json"
H5_PATH="$SCRATCH_BASE/lit_pcba_esm2_embeddings.h5"

echo "📂 LIT-PCBA Verileri SSD'ye kopyalanıyor..."
mkdir -p $LOCAL_SSD
# Sadece gerekli veri klasörlerini kopyalıyoruz
time rsync -aq $SCRATCH_BASE/ $LOCAL_SSD/

# 2. 🪄 JSON Yollarını SSD'ye Göre Güncelle
# Scratch yolunu LOCAL_SSD yoluyla değiştiriyoruz ki script dosyaları SSD'den okusun
sed "s|/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA|$LOCAL_SSD|g" $ORIG_JSON > $TEMP_JSON

echo "✅ JSON yolları SSD'ye uyarlandı: $TEMP_JSON"

# 3. Ortamı Hazırla
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 4. Çalıştırma (Final versiyonu çağırıyoruz)
echo "🚀 Benchmark Başlıyor: $(date)"

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Model/PLANET_eval_final.py \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2/PLANET_ESM2.iter-1419290 \
--out_dir $SCRATCH_BASE/benchmark_results_litpcba_ssd \
--h5_path $H5_PATH \
--json_path $TEMP_JSON \
--device cpu \
--batch_size 128 \
--feature_dims 512

# 5. Temizlik
echo "🧹 SSD temizleniyor..."
rm -rf $LOCAL_SSD
echo "İşlem bitti: $(date)"