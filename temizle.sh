#!/bin/bash
# --- PLANET_train_gpu.sh ---
#SBATCH -p akya-cuda         # GPU içeren kuyruk
#SBATCH --gres=gpu:1           # 1 GPU iste
#SBATCH --cpus-per-task=10   
#SBATCH --job-name=lr00005_planet_gpu_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G                    # 32G bellek (Gerekirse 64G'a çıkarılabilir)
#SBATCH --time=3-00:00:00            # 3 gün süre
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/planet_gpu_train_%j.err
#SBATCH -A hnaydin

rm -rf /tmp/hnaydin_planet_cache
echo "Temizlik tamamlandi, disk kontrol ediliyor:"
df -h /tmp
