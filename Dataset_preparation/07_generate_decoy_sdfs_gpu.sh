#!/bin/bash
# --- 07_generate_decoy_sdfs.sh ---
# Amaç: Sadece tek bir PDB ID'si üzerinde Cep Oluşturma ve BANANA skorlama testini çalıştırmak.
#SBATCH -p barbun-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=20  
#SBATCH -J 07_generate_decoy_sdfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=10:00:00            # 3 gün süre
#SBATCH -A hnaydin
#SBATCH -J 07_generate_decoy_sdfs
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_decoy_sdfs_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/generate_decoy_sdfs_%A.err



echo "Is basladi: $(date)"

source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

export NUM_CPUS=${SLURM_CPUS_PER_TASK:-40} 
PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/07_generate_decoy_sdfs_deneme.py"

# -p argümanı ile PDB ID'sini Python scriptine gönderiyoruz
python3 "$PYTHON_SCRIPT" -n $NUM_CPUS

echo "Is bitti: $(date)"
