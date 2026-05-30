#!/bin/bash
# --- 03_prepare_protein.sh ---
# Amaç: PDBFixer ile Protein Hazırlama ve Prepared PDB yollarını JSON haritasına kaydetme.

#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH -J 03_prepare_protein
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/prepare_protein_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/prepare_protein_%A.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=300G

echo "Is basladi: $(date)"

# 1. ORTAM AYARI
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 2. AYARLAR
export NUM_CPUS=${SLURM_CPUS_PER_TASK:-40}
PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/03_prepare_protein.py"

# KRİTİK: 02. adımdan gelen orijinal JSON yolu
export ORIGINAL_PK_MAP="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd.json"
# Yeni oluşturulacak JSON yolu:
export PREPARED_PK_MAP="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json"

# 3. PYTHON SCRIPTİNİ ÇALIŞTIRMA
echo "Protein hazırlama ve JSON güncelleme başlatılıyor..."

python3 "$PYTHON_SCRIPT" \
    --json_path "$ORIGINAL_PK_MAP" \
    --njobs $NUM_CPUS

if [ $? -ne 0 ]; then
    echo "❌ KRİTİK HATA: Protein hazırlama (PDBFixer) adımı başarısız oldu."
    exit 1
fi
echo "✅ JSON haritası güncellendi ve yeni dosyaya kaydedildi: $PREPARED_PK_MAP"

echo "Is bitti: $(date)"

# --- SIRALAMA GÜNCELLEMESİ ---
echo -e "\n--- SIRADAKİ ADIM (ADIM 4) ---"
echo "Sırada Ligand Özelliklerini Çıkarma (04_extract_binder_properties.sh) adımı var. Bu script, yeni JSON haritasını ($PREPARED_PK_MAP) kullanmalıdır."