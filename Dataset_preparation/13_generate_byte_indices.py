import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- AYARLAR ---
DATA_DIR = os.path.join(PLANET_ROOT, "datasets/all_pickles/")
N_JOBS = 32 # Arf cluster üzerinde paralel işlem için ideal

def reformat_and_index_decoys(file_name):
    """
    SADECE _decoys.pkl dosyalarını:
    1. Tek parça listeden kurtarır.
    2. Stream (nesne nesne) formatında yeniden yazar.
    3. Her nesnenin başladığı byte adresini (offset) kaydeder.
    """
    # Filtre: Sadece decoys dosyaları
    if not file_name.endswith('_decoys.pkl'):
        return None
        
    decoy_path = os.path.join(DATA_DIR, file_name)
    byte_path = decoy_path.replace('.pkl', '_byte.pkl')
    temp_path = decoy_path + ".tmp"
    
    try:
        # 1. Mevcut tüm decoyları RAM'e oku
        with open(decoy_path, 'rb') as f:
            data = pickle.load(f)
            
        # Eğer veri liste değilse bile uyum için listeye çevir
        if not isinstance(data, list):
            data = [data]
            
        offsets = []
        # 2. Dosyayı STREAM formatında (rastgele erişilebilir) yeniden yaz
        with open(temp_path, 'wb') as f_out:
            for obj in data:
                offset = f_out.tell() # O anki byte pozisyonunu al
                pickle.dump(obj, f_out, pickle.HIGHEST_PROTOCOL)
                offsets.append(offset)
                
        # 3. Geçici dosyayı orijinalin üzerine yaz
        os.replace(temp_path, decoy_path)
        
        # 4. Sadece tam sayılardan oluşan küçük byte indeksini oluştur
        with open(byte_path, 'wb') as f_idx:
            pickle.dump(offsets, f_idx)
            
        return "Success"
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"Error: {str(e)}"

def main():
    print(f"📂 Klasör taranıyor: {DATA_DIR}")
    all_files = os.listdir(DATA_DIR)
    # Sadece decoys.pkl olanları filtrele (binderlar elenir)
    target_files = [f for f in all_files if f.endswith('_decoys.pkl')]
    
    print(f"🚀 {len(target_files)} decoy dosyası işleniyor...")

    with Pool(processes=N_JOBS) as pool:
        results = list(tqdm(pool.imap_unordered(reformat_and_index_decoys, target_files), 
                            total=len(target_files), desc="İşleniyor"))

    success = [r for r in results if r == "Success"]
    print(f"\n✅ İşlem Tamamlandı: {len(success)} decoy dosyası indekslendi.")

if __name__ == "__main__":
    main()