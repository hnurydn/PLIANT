#!/bin/bash
#SBATCH -C weka
#SBATCH --partition=hamsi
#SBATCH --cpus-per-task=54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A hnaydin
#SBATCH -J 06_pocket_banana
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/logs/banana_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/logs/banana_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G

echo "Is basladi: $(date)"
echo "Calisilan Dugum: $SLURM_NODELIST"

# 1. ORTAM AYARI (Script içinde Conda'yı sağlıklı çalıştırmak için)
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate banana  # Ortam adını direkt kullanmak daha güvenlidir

# 2. KRİTİK ADIM: PYTHONPATH
# 'terrace' kütüphanesini bulamama sebebin bu olabilir. 
# Eğer kütüphane banana klasöründeyse orayı ekleyelim.
export PYTHONPATH=$PYTHONPATH:/arf/scratch/hnaydin/banana

# --- SEÇENEK A: SADECE TEST (1d4p için) ---
# GPU kuyruğunda olduğumuz için --no_gpu'yu kaldırdım.
python /arf/scratch/hnaydin/banana/inference.py \
/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/temp_smi_files/1d4p_candidates.smi \
/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/banana_pocket_pdbs_12A/1d4p_pocket.pdb \
--out_file /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/test_debug_scores.txt
--no_gpu
