#!/bin/bash
# --- 09_generate_ESM2_embeddings.sh ---
# Amaç: PDBbind veri setindeki tüm PDB ID'leri için ComplexPocket PKL dosyalarını oluşturmak.
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40  # Paralel süreç sayınızla eşleşmeli
#SBATCH -J 09_generate_ESM2_embeddings
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_ESM2_embeddings_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_ESM2_embeddings_%A.err
#SBATCH --time=1-00:00:00 
#SBATCH --mem=300G         # Örn: 40 GB bellek


echo "Is basladi: $(date)"
echo "Host: $(hostname)"

# Gerekli ortamı yükle
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# Script'in bulunduğu ana dizine git
cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/

echo "Calistirilacak Python script'i: 09_generate_ESM2_embeddings.py"

# Adım 2: ESM-2 Hesaplama script'ini çalıştır
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/09_generate_ESM2_embeddings.py

echo "Is bitti: $(date)"

