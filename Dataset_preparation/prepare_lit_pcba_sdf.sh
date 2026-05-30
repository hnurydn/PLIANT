#!/bin/bash
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J sdf_generation
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/sdf_gen_%j.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/sdf_gen_%j.err
#SBATCH --time=24:00:00 
#SBATCH --mem=100G

# 1. Ortami Dogru Sekilde Yukle (Hata veren ~/.bashrc yerine bunu kullaniyoruz)
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh

# 2. Conda Ortamini Aktif Et
conda activate pocketgen

echo "🚀 Islem Baslatiliyor..."
echo "Python Yolu: $(which python)"

# 3. Python Scriptini Calistir (-u parametresi loglari anlik gosterir)
python -u /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/prepare_lit_pcba_sdf.py
