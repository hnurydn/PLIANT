#!/bin/bash
# --- 09_generate_ESM2_embeddings_crossdocked.sh ---
# Amaç: PDBbind veri setindeki tüm PDB ID'leri için ComplexPocket PKL dosyalarını oluşturmak.
#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -J 09_generate_ESM2_embeddings_crossdocked
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_ESM2_embeddings_crossdocked_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_ESM2_embeddings_crossdocked_%A.err


echo "Is basladi: $(date)"
echo "Host: $(hostname)"

# Gerekli ortamı yükle
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# Script'in bulunduğu ana dizine git
cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/

echo "Calistirilacak Python script'i: 09_generate_ESM2_embeddings_crossdocked.py"

# Adım 2: ESM-2 Hesaplama script'ini çalıştır
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/09_generate_ESM2_embeddings_crossdocked.py \
    --json_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_new_records.json \
    --extracted_dir /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/crossdocked_extracted \
    --h5_path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pdbbind2020_esm2_embeddings_v2.h5 \
    --batch_size 2

echo "Is bitti: $(date)"

