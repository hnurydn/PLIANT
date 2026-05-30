#!/bin/bash
# --- PLANET_eval_ESM2_cpu.sh ---
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
SCRATCH_DUDE="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude"
LOCAL_SSD="/tmp/${USER}_dude_eval"
ORIG_JSON="$SCRATCH_DUDE/dude_paths_prepared_arf.json"
TEMP_JSON="$LOCAL_SSD/dude_paths_ssd.json"

echo "📂 Veriler SSD'ye kopyalanıyor..."
mkdir -p $LOCAL_SSD
# Sadece gerekli klasörü (targetları) kopyalayalım
rsync -aq $SCRATCH_DUDE/ $LOCAL_SSD/

# 2. 🪄 JSON Yollarını SSD'ye Göre Güncelle (Sed ile sihir yapıyoruz)
# Scratch yolunu LOCAL_SSD yoluyla değiştiriyoruz
sed "s|/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude|$LOCAL_SSD|g" $ORIG_JSON > $TEMP_JSON

echo "✅ JSON yolları SSD'ye uyarlandı."

# 3. Çalıştırma
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE_benchmark_DEBUG.py \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRANING_MODEL_ESM2/PLANET_ESM2.iter-1419290 \
--out_dir $SCRATCH_DUDE/benchmark_results_ssd \
--h5_path $SCRATCH_DUDE/dude_esm2_embeddings.h5 \
--device cpu \
--feature_dims 512

# 4. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"