#!/bin/bash
# --- 08_prepare_planet_data_deneme.sh ---
# Amaç: PDBbind veri setindeki tüm PDB ID'leri için ComplexPocket PKL dosyalarını oluşturmak.
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40  # Paralel süreç sayınızla eşleşmeli
#SBATCH -J 08_prepare_planet_data_deneme
#SBATCH -o /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/prepare_planet_data_deneme_%A.out
#SBATCH -e /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/prepare_planet_data_deneme_%A.err
#SBATCH --time=1-00:00:00 
#SBATCH --mem=40G         # Örn: 40 GB bellek

# 1. ORTAM AYARI
source ~/.bashrc
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

# 2. AYARLAR
PYTHON_SCRIPT="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation/process_PDBBind_new.py"

# KRİTİK AYAR: PDBbind'in YIL klasörlerini içeren en üst DİZİNİ!
# Örn: /pdbbind2020 dizini altında 1981-2000, 2001-2010 vb. olmalıdır.
PDB_ROOT_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020" 

# Paralel süreç sayısı (cpus-per-task ile aynı olmalıdır)
N_JOBS=40

# 3. PYTHON SCRIPTİNİ ÇALIŞTIRMA

echo "TÜM PDBBIND VERİ SETİ İÇİN PKL OLUŞTURMA BAŞLATILIYOR."
echo "Kök Dizin: $PDB_ROOT_DIR"
echo "Paralel Süreç Sayısı (njobs): $N_JOBS"

# Python scriptine PDBbind'in en üst dizinini (-d) ve paralel süreç sayısını (-n) gönderiyoruz.
# Python scripti bu dizini tarayarak tüm PDB'leri bulacaktır.
python3 $PYTHON_SCRIPT -d "$PDB_ROOT_DIR" -n $N_JOBS

if [ $? -ne 0 ]; then
    echo "❌ KRİTİK HATA: ComplexPocket PKL oluşturma başarısız oldu. Ayrıntılar için .err dosyasını kontrol edin."
    exit 1
fi

echo "✅ TÜM İŞLEM TAMAMLANDI."
echo "PKL dosyaları ilgili PDB ID klasörlerinde oluşturulmuştur."