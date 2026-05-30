#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J 20_prepare_lit_pcba
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/20_prepare_lit_pcba_%j.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/20_prepare_lit_pcba_%j.err
#SBATCH --time=1-00:00:00 
#SBATCH --mem=100G

# --- KRİTİK DÜZELTME ---
# .bashrc yerine Conda'yı doğrudan kaynak dosyasından yüklüyoruz.
# Bu yöntem SLURM üzerinde hatasız çalışır.
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh

# Ortamı aktif et
conda activate pocketgen

echo "✅ Ortam Hazır."
echo "Python Yolu: $(which python)"
echo "🚀 LIT-PCBA PKL Oluşturma Başlıyor..."

# -u parametresi logları anında dosyaya yazar, bekleme yapmaz.
python -u /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/20_prepare_lit_pcba_pkl.py
