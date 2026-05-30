import os
import json
import pickle
import sys
from multiprocessing import Pool
from tqdm import tqdm
# --- Kütüphane Yükleme ---
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.warning')
    # Scriptin olduğu yerde chemutils.py olmalı
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from chemutils import ComplexPocket, Mol
except ImportError as e:
    print(f"❌ HATA: Gerekli kütüphaneler eksik: {e}")
    sys.exit(1)

# ==========================================
#              AYARLAR
# ==========================================
# Referans JSON dosyan
JSON_PATH = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'

# Çekirdek Sayısı (Sunucuyu yormadan hızlı bitmesi için)
N_JOBS = 20
# ==========================================

def process_record(item):
    """
    JSON'dan gelen PDB kaydını işler:
    1. Protein yolundan klasörü bulur.
    2. O klasörde _decoys.pkl var mı bakar.
    3. Varsa okur, byte'a çevirir ve _decoys_byte.pkl olarak kaydeder.
    """
    pdb_id, data = item
    
    try:
        # JSON'dan protein yolunu al:
        # Örn: /media/.../1981-2000/2ypi/2ypi_protein_prepared.pdb
        protein_path = data.get('protein_pdb')
        
        if not protein_path:
            return f"SKIP: Protein yolu yok ({pdb_id})"
            
        # PDB klasörünün kendisini bul (/media/.../1981-2000/2ypi/)
        pdb_folder = os.path.dirname(protein_path)
        
        # Hedeflenen Decoy Dosyası Yolları
        original_decoy_path = os.path.join(pdb_folder, f"{pdb_id}_decoys.pkl")
        target_byte_path = os.path.join(pdb_folder, f"{pdb_id}_decoys_byte.pkl")
        
        # 1. Kaynak dosya var mı?
        if not os.path.exists(original_decoy_path):
            return f"SKIP: Decoy PKL bulunamadı ({pdb_id})"
            
        # 2. Hedef dosya zaten var mı? (Resume özelliği)
        if os.path.exists(target_byte_path) and os.path.getsize(target_byte_path) > 100:
            return 0 # Zaten yapılmış, atla.

        # 3. Yükle -> Dönüştür -> Kaydet
        with open(original_decoy_path, 'rb') as f:
            content = pickle.load(f)
            
        if not isinstance(content, list) or len(content) == 0:
            return f"ERR: Boş veya hatalı liste ({pdb_id})"
            
        # Byte Dönüşümü
        byte_list = []
        for obj in content:
            if isinstance(obj, bytes):
                byte_list.append(obj) # Zaten byte ise dokunma
            else:
                # Nesneyi dondur (Serialize)
                blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                byte_list.append(blob)
                
        # Kaydet
        with open(target_byte_path, 'wb') as f:
            pickle.dump(byte_list, f, pickle.HIGHEST_PROTOCOL)
            
        return 1 # Başarılı dönüşüm

    except Exception as e:
        return f"HATA: {pdb_id} - {str(e)}"

def main():
    print(f"--- DECOY BYTE DÖNÜŞTÜRÜCÜ (YERİNDE DÖNÜŞÜM) ---")
    print(f"Kaynak JSON: {JSON_PATH}")
    
    # JSON Yükle
    if not os.path.exists(JSON_PATH):
        print("❌ HATA: JSON dosyası bulunamadı!")
        sys.exit(1)
        
    with open(JSON_PATH, 'r') as f:
        data_map = json.load(f)
        
    pdb_items = list(data_map.items())
    print(f"Toplam PDB Kaydı: {len(pdb_items)}")
    print(f"🚀 İşlem başlıyor ({N_JOBS} çekirdek)...")
    
    # Havuz (Pool) ile paralel işlem
    with Pool(N_JOBS) as pool:
        results = list(tqdm(pool.imap_unordered(process_record, pdb_items), 
                            total=len(pdb_items), 
                            desc="Dönüştürülüyor"))
        
    # İstatistik
    success = sum(1 for r in results if r == 1)
    skipped = sum(1 for r in results if r == 0) # Zaten varsa
    errors = [r for r in results if isinstance(r, str) and (r.startswith("ERR") or r.startswith("HATA"))]
    missing = [r for r in results if isinstance(r, str) and r.startswith("SKIP")]
    
    print("\n" + "="*30)
    print(f"✅ Yeni Oluşturulan: {success}")
    print(f"⏩ Zaten Mevcut (Atlanan): {skipped}")
    print(f"🚫 Decoy Dosyası Olmayanlar: {len(missing)}")
    print(f"❌ Hatalar: {len(errors)}")
    
    if errors:
        print("\nİlk 5 Hata:")
        for e in errors[:5]: print(e)

if __name__ == "__main__":
    main()