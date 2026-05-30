#!/bin/bash
# --- 09_generate_ESM2_embeddings_dude.sh ---
# Amaç: PDBbind veri setindeki tüm PDB ID'leri için ComplexPocket PKL dosyalarını oluşturmak.
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH -J 09_generate_ESM2_embeddings_litpcba
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_ESM2_embeddings_lit_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_ESM2_embeddings_lit_%A.err


echo "🚀 ESM-2 Embedding Üretimi Başladı: $(date)"

# 1. Ortamı Aktif Et
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 2. Dosya Yolları
# Az önce oluşturduğumuz FULL FASTA dosyan
FASTA_IN="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA/lit_pcba_sequences_full.fasta"
# Oluşacak yeni H5 dosyası (Benchmark script'ine bu yolu vereceğiz)
H5_OUT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA/lit_pcba_esm2_embeddings.h5"

# 3. Üretimi Başlat
# --fasta ve --output argümanlarını script'ine uygun şekilde geçiyoruz
python /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/09_generate_ESM2_embeddings_dude.py \
    --fasta $FASTA_IN \
    --output $H5_OUT

echo "✅ ESM-2 Üretimi Tamamlandı: $(date)"

