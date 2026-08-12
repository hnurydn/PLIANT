import json
import pandas as pd
import sys
import os
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- Dosya Yolları ---
POCKET_MAP_JSON = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json")
DECOY_CSV = os.path.join(PLANET_ROOT, "datasets/decoy_candidates_find_100_suitable.csv")

def find_missing_ids(json_path, csv_path):
    """
    JSON dosyasındaki anahtarlar (PDB ID'ler) ile CSV dosyasındaki 
    binder_pdb_id sütununu karşılaştırır ve CSV'de olmayanları listeler.
    """
    try:
        # 1. JSON Dosyasını Yükle ve Anahtarları Çek
        with open(json_path, 'r') as f:
            pdb_map = json.load(f)
        
        # Anahtarlar (JSON'daki tüm PDB ID'leri)
        json_ids = set(pdb_map.keys())
        
        # 2. CSV Dosyasını Yükle ve PDB ID'leri Çek
        # Sadece ilgili sütunu yükleyerek bellek kullanımını optimize ediyoruz
        df_decoy = pd.read_csv(csv_path, usecols=['binder_pdb_id'])
        
        # CSV'deki benzersiz PDB ID'leri
        csv_ids = set(df_decoy['binder_pdb_id'].unique())
        
        # 3. Karşılaştırma: JSON'da Olup CSV'de Olmayanları Bul
        # Bu işlem (A - B) küme farkını hesaplar.
        missing_in_csv = json_ids - csv_ids
        
        print(f"--- SONUÇLAR ---")
        print(f"JSON'daki Toplam Benzersiz PDB ID: {len(json_ids)}")
        print(f"CSV'deki Toplam Benzersiz PDB ID: {len(csv_ids)}")
        print("-" * 20)
        
        if missing_in_csv:
            print(f"❌ CSV'de Eksik Olan PDB ID Sayısı (JSON'da mevcut): {len(missing_in_csv)}")
            print(f"\nEksik ID Listesi:")
            # Listeyi alfabetik sırayla yazdır
            for id_ in sorted(list(missing_in_csv)):
                print(id_)
        else:
            print("✅ JSON'daki tüm PDB ID'leri CSV dosyasında mevcuttur.")
            
    except FileNotFoundError:
        print("HATA: Dosya yolu bulunamadı. Lütfen yolları kontrol edin.", file=sys.stderr)
    except KeyError:
        print("HATA: CSV dosyasında 'binder_pdb_id' sütunu bulunamadı.", file=sys.stderr)
    except Exception as e:
        print(f"Beklenmeyen Hata: {e}", file=sys.stderr)

if __name__ == '__main__':
    find_missing_ids(POCKET_MAP_JSON, DECOY_CSV)