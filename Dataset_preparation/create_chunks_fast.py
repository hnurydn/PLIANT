'''import os, pickle, torch, sys, json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# --- 1. MODÜL YÖNLENDİRMESİ ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import chemutils_esm2_vector as chemutils
sys.modules['chemutils'] = chemutils

# --- 2. YOLLAR ---
INDEX_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/index/")
ESM2_STORAGE_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_esm2_pkl")
PK_MAP_JSON = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json")

# YENİ HEDEF KLASÖRÜMÜZ (Saf Tensörler Burada Birikecek)
TENSOR_OUT_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_tensors")
os.makedirs(TENSOR_OUT_DIR, exist_ok=True)

def process_pdb(args):
    pdb, original_pk, binder_path, decoy_path = args
    out_file = os.path.join(TENSOR_OUT_DIR, f"{pdb}_tensors.pt")
    
    # Zaten dönüştürülmüşse atla (Kaldığı yerden devam edebilmesi için)
    if os.path.exists(out_file): return True

    try:
        pdb_dict = {
            'pdb_id': pdb,
            'pK': original_pk,
            'decoys': []
        }

        # --- A. BINDER (GERÇEK LİGAND) İŞLEMLERİ ---
        with open(binder_path, 'rb') as f:
            binder_obj = pickle.load(f)
            binder_obj.pK = original_pk

        # Orijinal fonksiyonu kullanarak saf tensörleri elde et (decoy_flag=False ile tekil işlem)
        res_binder = chemutils.tensorize_all([binder_obj], decoy_flag=False)
        
        # res: res_f, mol_f, mol_int, pl_int, pKs, pK_flags, complex_labels
        pdb_dict['res_f'] = res_binder[0]          # Cep (Pocket) matrisleri (SADECE BURADA KAYDEDİLİR)
        pdb_dict['binder_mol_f'] = res_binder[1]   # Binder'ın atom/bağ graf matrisleri
        pdb_dict['binder_mol_int'] = res_binder[2] 
        pdb_dict['binder_pl_int'] = res_binder[3]

        # --- B. DECOY (YANILTICI) İŞLEMLERİ ---
        if os.path.exists(decoy_path):
            with open(decoy_path, 'rb') as f:
                while True:
                    try:
                        decoy_obj = pickle.load(f)
                        decoy_obj.pK = 0.0
                        
                        # Eksiklik olmaması adına ESM özelliklerini binder'dan garantile
                        decoy_obj.res_features = binder_obj.res_features
                        decoy_obj.pocket_residues = binder_obj.pocket_residues

                        res_decoy = chemutils.tensorize_all([decoy_obj], decoy_flag=False)
                        
                        decoy_dict = {
                            'mol_f': res_decoy[1],     # Decoy'un graf matrisi
                            'mol_int': res_decoy[2],
                            # Not: pl_int decoy'larda her zaman sıfır (0) olduğu için 
                            # diskte yer kaplamasın diye kaydetmiyoruz. DataLoader'da 0 basacağız!
                        }
                        pdb_dict['decoys'].append(decoy_dict)
                    except EOFError:
                        break # 540 tane bitince döngüden çık

        # Saf tensör sözlüğünü diske ışık hızında kaydet
        torch.save(pdb_dict, out_file)
        return True
    except Exception as e:
        print(f"\nHATA ({pdb}): {e}")
        return False

def main():
    # Eğitim setlerindeki tüm benzersiz PDB'leri topla
    sets = ['TrainSet_Strict_Cleaned.csv', 'ValidSet.csv', 'CoreSet.csv']
    all_tasks = []
    processed_pdbs = set()

    print("CSV Dosyaları Taranıyor...")
    for csv_name in sets:
        csv_path = os.path.join(INDEX_DIR, csv_name)
        if not os.path.exists(csv_path): continue
        
        df = pd.read_csv(csv_path)
        pdb_col = 'PDB_code' if 'PDB_code' in df.columns else 'pdb_id'
        
        for _, row in df.iterrows():
            pdb = str(row[pdb_col]).lower()
            if pdb in processed_pdbs: continue
            
            binder_path = os.path.join(ESM2_STORAGE_DIR, f"{pdb}_binder_esm512d.pkl")
            decoy_path = os.path.join(ESM2_STORAGE_DIR, f"{pdb}_decoys_esm512d.pkl")
            
            if os.path.exists(binder_path):
                all_tasks.append((pdb, float(row['pK']), binder_path, decoy_path))
                processed_pdbs.add(pdb)

    print(f"Toplam Dönüştürülecek PDB Sayısı: {len(all_tasks)}")
    print("Tensörleştirme ve Paketleme Başlıyor (16 Çekirdek Aktif)...")

    # Çoklu işlemci ile turbo hızda dönüştür
    basarili = 0
    with Pool(16) as p:
        for durum in tqdm(p.imap_unordered(process_pdb, all_tasks), total=len(all_tasks)):
            if durum: basarili += 1

    print(f"\n✅ İŞLEM TAMAM! {basarili}/{len(all_tasks)} PDB başarıyla .pt tensörlerine çevrildi.")
    print(f"Yeni Saf Tensörleriniz '{TENSOR_OUT_DIR}' klasöründe sizi bekliyor.")

if __name__ == "__main__":
    main()'''
    
'''import os, torch, pickle
from tqdm import tqdm

# --- YOLLAR ---
TENSOR_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_tensors")
CHUNK_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_tensor_chunks")
INDEX_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/index")

os.makedirs(CHUNK_DIR, exist_ok=True)

def pack_set(pkl_name, prefix):
    index_path = os.path.join(INDEX_DIR, pkl_name)
    if not os.path.exists(index_path): 
        print(f"⚠️ {pkl_name} bulunamadı, atlanıyor.")
        return
        
    with open(index_path, 'rb') as f:
        data_list = pickle.load(f)
        
    # Bu sete ait benzersiz PDB'leri bul
    pdb_set = set()
    for record in data_list:
        pdb_code = os.path.basename(record[0]).split('_')[0]
        pdb_set.add(pdb_code)
        
    print(f"\n📦 [{prefix.upper()}] Seti Paketleniyor... (Toplam Benzersiz PDB: {len(pdb_set)})")
    
    CHUNK_SIZE = 1000
    current_chunk = {}
    chunk_idx = 0
    
    # Sadece listedeki PDB'leri depodan çek ve paketle
    for pdb in tqdm(pdb_set):
        t_file = os.path.join(TENSOR_DIR, f"{pdb}_tensors.pt")
        if os.path.exists(t_file):
            current_chunk[pdb] = torch.load(t_file)
            
        if len(current_chunk) == CHUNK_SIZE:
            out_path = os.path.join(CHUNK_DIR, f"{prefix}_chunk_{chunk_idx:03d}.pt")
            torch.save(current_chunk, out_path)
            current_chunk = {} # Hafızayı boşalt
            chunk_idx += 1
            
    # Geriye kalan küsuratı son paket olarak yaz
    if len(current_chunk) > 0:
        out_path = os.path.join(CHUNK_DIR, f"{prefix}_chunk_{chunk_idx:03d}.pt")
        torch.save(current_chunk, out_path)
        print(f"✅ {prefix} setinin tüm paketleri diske yazıldı.")

def main():
    print("Mevcut tensörler setlere (Train/Valid/Test) göre ayrıştırılarak paketleniyor...")
    
    pack_set("train_esm2.pkl", "train")
    pack_set("valid_esm2.pkl", "valid")
    pack_set("core_esm2.pkl", "test")

    print("\n🚀 TÜM İŞLEMLER TAMAM! Ayrıştırılmış dev paketleriniz 'planet_tensor_chunks' klasöründe hazır.")

if __name__ == "__main__":
    main()'''
    
"""import os, torch, math
from tqdm import tqdm

SOURCE_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_tensors/train")
SAVE_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_tensors/shards")
os.makedirs(SAVE_DIR, exist_ok=True)

files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.pt')]
num_shards = 4
chunk_size = math.ceil(len(files) / num_shards)

print(f"📦 {len(files)} dosya {num_shards} parçaya bölünüyor (Her parça ~{chunk_size} dosya)...")

for i in range(num_shards):
    shard_data = {}
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(files))
    shard_files = files[start_idx:end_idx]
    
    print(f"\n📂 Parça {i+1}/{num_shards} hazırlanıyor...")
    for f in tqdm(shard_files):
        pdb_id = f.replace('_tensors.pt', '')
        shard_data[pdb_id] = torch.load(os.path.join(SOURCE_DIR, f), map_location='cpu')
    
    save_path = os.path.join(SAVE_DIR, f"TRAIN_SHARD_{i}.pt")
    print(f"💾 Parça {i} kaydediliyor: {save_path}")
    torch.save(shard_data, save_path)
    
    # 🧹 KRİTİK: RAM'i boşaltmak için veriyi sil
    del shard_data
    import gc
    gc.collect()

print("\n✅ Tüm parçalar başarıyla oluşturuldu!")"""

# fix_missing_valid_tensors.py
import os, pickle, torch, sys, json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import chemutils_esm2_vector as chemutils
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout


sys.modules['chemutils'] = chemutils

ESM2_DIR  = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_esm2_pkl")
OUT_DIR   = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/planet_tensors/valid")
INDEX_CSV = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/index/ValidSet.csv")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INDEX_CSV)
pk_dict = dict(zip(df['PDB_code'].str.lower(), df['pK']))

# Tüm valid pdb'leri işle (varsa atla)
all_pdbs = list(pk_dict.keys())
print(f"Toplam {len(all_pdbs)} valid PDB işlenecek...")

def process_pdb(pdb):
    out_file = os.path.join(OUT_DIR, f"{pdb}_tensors.pt")
    if os.path.exists(out_file):
        return True  # Zaten var, atla

    binder_path = os.path.join(ESM2_DIR, f"{pdb}_binder_esm512d.pkl")
    decoy_path  = os.path.join(ESM2_DIR, f"{pdb}_decoys_esm512d.pkl")

    if not os.path.exists(binder_path):
        return False

    try:
        pdb_dict = {'pdb_id': pdb, 'pK': pk_dict.get(pdb, 0.0), 'decoys': []}

        with open(binder_path, 'rb') as f:
            binder_obj = pickle.load(f)
        binder_obj.pK = pdb_dict['pK']

        res_binder = chemutils.tensorize_all([binder_obj], decoy_flag=False)
        pdb_dict['res_f']          = res_binder[0]
        pdb_dict['binder_mol_f']   = res_binder[1]
        pdb_dict['binder_mol_int'] = res_binder[2]
        pdb_dict['binder_pl_int']  = res_binder[3]

        if os.path.exists(decoy_path):
            with open(decoy_path, 'rb') as f:
                while True:
                    try:
                        decoy_obj = pickle.load(f)
                        decoy_obj.pK = 0.0
                        decoy_obj.res_features    = binder_obj.res_features
                        decoy_obj.pocket_residues = binder_obj.pocket_residues
                        res_decoy = chemutils.tensorize_all([decoy_obj], decoy_flag=False)
                        pdb_dict['decoys'].append({'mol_f': res_decoy[1], 'mol_int': res_decoy[2]})
                    except EOFError:
                        break

        torch.save(pdb_dict, out_file)
        return True
    except Exception as e:
        print(f"HATA ({pdb}): {e}")
        return False

with Pool(16) as p:
    results = list(tqdm(p.imap_unordered(process_pdb, all_pdbs), total=len(all_pdbs)))

print(f"✅ Tamamlandı: {sum(results)}/{len(all_pdbs)}")