#!/bin/bash

# --- 01_prepare_datasets.sh ---
# Amaç: Gerekli tüm veri setlerini indirmek, PDBbind'i açmak ve dizin yapısını oluşturmak.

# 1. AYARLAR
# ------------------------------------------------------------------
# İndirilen ve açılan tüm verinin ana konumu (Sizin yeni yolunuz)
export BASE_DIR="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets"
# PDBbind ZIP dosyasının adı
PDBBIND_ZIP="pdbbind2020.zip"
# ------------------------------------------------------------------

echo "--- 1. VERİ İNDİRME VE HAZIRLIK BAŞLATILIYOR ---"
mkdir -p "$BASE_DIR"
echo "Hedef Dizin: $BASE_DIR"

# --- A. İNDİRME İŞLEMLERİ ---

# 1.1 PDBbind 2020 Veri Seti (Pozitif Örnekler)
echo -e "\n[1/5] PDBbind 2020 Veri Seti indiriliyor..."
wget -O "pdbbind2020.tar.gz" "https://pdbbind-1301734146.cos.ap-shanghai.myqcloud.com/v2020R1/P-L.tar.gz?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDzoOinb9RTFkyvc3D6j5AxVmmVyAyVllV%26q-sign-time%3D1762814752%3B1762818412%26q-key-time%3D1762814752%3B1762818412%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D089160e4584d1e71753f1251963f89e960b07686"
[ $? -eq 0 ] && echo "✅ PDBbind 2020 başarıyla indirildi." || echo "❌ HATA: PDBbind 2020 indirme hatası."

# 1.2 ChEMBL SDF Veri Seti (Decoy Kaynağı)
echo -e "\n[2/5] ChEMBL 29 SDF indiriliyor..."
wget -O "$BASE_DIR/chembl_29.sdf.gz" "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29.sdf.gz"
[ $? -eq 0 ] && echo "✅ ChEMBL 29 başarıyla indirildi." || echo "❌ HATA: ChEMBL 29 indirme hatası."

# 1.3 CASF-2016 Core Set ID'leri (Resmi Test Seti ID'leri)
echo -e "\n[3/5] CASF-2016 Core Set ID'leri indiriliyor..."
wget -O "$BASE_DIR/coreset.tar.gz" "https://github.com/S-Y-Z-Y-G/CASF-2016-Data/raw/master/coreset.tar.gz"
[ $? -eq 0 ] && echo "✅ CASF-2016 coreset başarıyla indirildi." || echo "❌ HATA: CASF-2016 coreset indirme hatası."

# 1.4 DUDE Decoy Kaynağı (İsteğe bağlı)
echo -e "\n[4/5] DUDE tüm alt kümeleri indiriliyor..."
wget -O "$BASE_DIR/dude_all.tar.gz" "https://dude.docking.org/db/subsets/all/all.tar.gz"
[ $? -eq 0 ] && echo "✅ DUDE başarıyla indirildi." || echo "❌ HATA: DUDE indirme hatası."

# 1.5 LIT-PCBA Decoy Kaynağı (İsteğe bağlı)
echo -e "\n[5/5] LIT-PCBA indiriliyor..."
wget -O "$BASE_DIR/LIT-PCBA.tar.gz" "https://zenodo.org/records/4019848/files/LIT-PCBA.tar.gz"
[ $? -eq 0 ] && echo "✅ LIT-PCBA başarıyla indirildi." || echo "❌ HATA: LIT-PCBA indirme hatası."

echo -e "\n--- B. PDBbind ARŞİV AÇMA İŞLEMİ ---"

tar -xzf pdbbind2020.tar.gz

wget -O index.tar.gz "https://pdbbind-1301734146.cos.ap-shanghai.myqcloud.com/v2020R1/index.tar.gz?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDzoOinb9RTFkyvc3D6j5AxVmmVyAyVllV%26q-sign-time%3D1762814752%3B1762818412%26q-key-time%3D1762814752%3B1762818412%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3Dc016fa83bd3dd9ce2d73c2b3a7abf7edaa2bce75"
tar -xzf  index.tar.gz
if [ $? -eq 0 ]; then
    echo "✅ PDBbind arşivi başarıyla $BASE_DIR içine açıldı."
else
    echo "❌ HATA: PDBbind dosyasını açarken bir sorun oluştu."
    exit 1
fi

# 4. KRİTİK İNDEX KONTROLÜ
INDEX_PATH="$BASE_DIR/index/INDEX_general_PL.2020R1.lst"
if [ -f "$INDEX_PATH" ]; then
    echo "✅ İndeks dosyası ($INDEX_PATH) başarıyla açıldı."
else
    echo "⚠️ UYARI: İndeks dosyası bulunamadı. Lütfen ZIP içeriğini kontrol edin."
fi

echo -e "\n--- VERİ İNDİRME VE PDBBIND HAZIRLIĞI TAMAMLANDI ---\n"