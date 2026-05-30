import pandas as pd
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm # İlerleme çubuğu

# --- AYARLAR ---
# Adım 2'den gelen ana JSON haritası (küçük harfli PDB ID'leri içerir)
JSON_MAP_FILE = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json"

# PDBbind'in resmi Core Set (CASF-2016) PDB ID'lerinin listesi (büyük harfli PDB ID'leri içerir)
CORE_SET_LIST_FILE = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/casf2016_input_PLANET.csv"

# Oluşturulacak CSV'lerin kaydedileceği yer
OUTPUT_CSV_DIR = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index"
# --- BİTİŞ AYARLARI ---

def main():
    print(f"Ana JSON haritası okunuyor: {JSON_MAP_FILE}")
    with open(JSON_MAP_FILE, 'r') as f:
        full_data_map = json.load(f)
    
    # JSON'daki tüm ID'ler zaten küçük harfli (örn: "1zsb")
    all_pdb_ids = set(full_data_map.keys())
    print(f"Toplam {len(all_pdb_ids)} PDB ID'si bulundu (küçük harfli).")

    # 1. Core Set'i (Test Set) Yükle
    try:
        core_df = pd.read_csv(CORE_SET_LIST_FILE)
        
        # --- DÜZELTME BURADA ---
        # Sütun adı 'pdb_id' (dosyanızdan doğrulandı)
        # ve .str.lower() ile tüm ID'leri küçük harfe çeviriyoruz.
        core_pdb_ids = set(core_df['pdb_id'].str.lower()) 
        # -----------------------
        
        print(f"CASF-2016 (Core Set) için {len(core_pdb_ids)} PDB ID'si bulundu ve küçük harfe çevrildi.")
    except Exception as e:
        print(f"HATA: Core set CSV ({CORE_SET_LIST_FILE}) okunamadı: {e}")
        print("CSV dosyasının 'pdb_id' sütununa sahip olduğundan emin olun.")
        return

    # 2. Train/Valid setini oluştur (Core settekiler HARİÇ)
    # Artık ikisi de küçük harfli olduğu için bu çıkarma işlemi doğru çalışacak
    train_valid_ids = list(all_pdb_ids - core_pdb_ids)
    print(f"Core set çıkarıldıktan sonra {len(train_valid_ids)} PDB ID'si Train/Valid için kaldı.")

    # 3. %90 Train / %10 Valid olarak ayır (Makaledeki gibi)
    train_ids, valid_ids = train_test_split(train_valid_ids, test_size=0.1, random_state=42)
    print(f"Eğitim seti (TrainSet): {len(train_ids)} PDB ID")
    print(f"Validasyon seti (ValidSet): {len(valid_ids)} PDB ID")
    
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

    # 4. DataFrame'leri oluştur ve CSV olarak kaydet
    
    def create_df(pdb_id_list):
        records = []
        for pdb_id in tqdm(pdb_id_list, desc="PDB ID'leri işleniyor"):
            # pdb_id'ler zaten küçük harfli
            if pdb_id in full_data_map:
                records.append({
                    "PDB_code": pdb_id.lower(), 
                    # DÜZELTME BURADA YAPILDI: 'pKd' yerine sizin kullandığınız 'pK_d' kullanıldı.
                    "pK": full_data_map[pdb_id].get('pK_d', np.nan),
                    "type": "origin" # Tüm pozitiflerimiz 'origin' tipindedir
                })
        return pd.DataFrame(records)

    # TrainSet.csv
    print("TrainSet.csv oluşturuluyor...")
    train_df = create_df(train_ids)
    train_df.to_csv(os.path.join(OUTPUT_CSV_DIR, "TrainSet.csv"))
    print(f"TrainSet.csv dosyası {OUTPUT_CSV_DIR} içine kaydedildi.")

    # ValidSet.csv
    print("ValidSet.csv oluşturuluyor...")
    valid_df = create_df(valid_ids)
    valid_df.to_csv(os.path.join(OUTPUT_CSV_DIR, "ValidSet.csv"))
    print(f"ValidSet.csv dosyası {OUTPUT_CSV_DIR} içine kaydedildi.")

    # CoreSet.csv
    print("CoreSet.csv oluşturuluyor...")
    core_df_final = create_df(core_pdb_ids)
    core_df_final.to_csv(os.path.join(OUTPUT_CSV_DIR, "CoreSet.csv"))
    print(f"CoreSet.csv dosyası {OUTPUT_CSV_DIR} içine kaydedildi.")
    
    print("\nCSV bölüştürme (splitting) işlemi tamamlandı.")

if __name__ == "__main__":
    main()