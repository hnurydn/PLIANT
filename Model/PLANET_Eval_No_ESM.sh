#!/bin/bash
# --- PLANET_eval_ESM2.sh ---
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20 
#SBATCH --job-name=planet_esm2_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G
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
rsync -aq $SCRATCH_DUDE/ $LOCAL_SSD/

# 2. JSON Yollarını SSD'ye Göre Güncelle
sed "s|/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude|$LOCAL_SSD|g" $ORIG_JSON > $TEMP_JSON

echo "✅ JSON yolları SSD'ye uyarlandı."

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 3. Model 1: ESM_UW Çalıştırma
echo "🚀 Model 1: No_ESM_UW Başlatılıyor..."
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE_benchmark_No_ESM.py \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_NoUW/PLANET.iter-1940000 \
--out_dir $SCRATCH_DUDE/benchmark_results \
--out_name benchmark_No_ESM_NoUW \
--feature_dims 300

# 4. Model 2: ESM_NoUW Çalıştırma
echo "🚀 Model 2:  No_ESM_NoUW Başlatılıyor..."
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/DUDE_benchmark_No_ESM.py \
--model_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/PLANET_TRAINING_MODEL_NoESM_UW/PLANET.iter-1940000 \
--out_dir $SCRATCH_DUDE/benchmark_results \
--out_name benchmark_No_ESM_UW \
--feature_dims 300

# 5. Temizlik
rm -rf $LOCAL_SSD
echo "🧹 SSD temizlendi, işlem tamam!"