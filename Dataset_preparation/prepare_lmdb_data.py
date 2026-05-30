import lmdb
import os
import pickle
from tqdm import tqdm

def create_lmdb(data_path, lmdb_path):
    # Yaklaşık 500GB yer ayırıyoruz (HPC'de disk kotana dikkat et)
    env = lmdb.open(lmdb_path, map_size=int(500e9)) 
    
    # Tüm alt klasörleri tara (1981-2000, 2011-2019 vb.)
    folders = []
    for root, dirs, files in os.walk(data_path):
        if any(f.endswith('_binder.pkl') for f in files):
            folders.append(root)

    print(f"Toplam {len(folders)} PDB klasörü bulundu. Kasaya aktarılıyor...")

    with env.begin(write=True) as txn:
        for folder in tqdm(folders):
            pdb_id = os.path.basename(folder) # Örn: '186l'
            
            # Binder ve Decoy yollarını bul
            binder_p = os.path.join(folder, f"{pdb_id}_binder.pkl")
            # Öncelik senin hazırladığın 'byte' dosyasında!
            decoy_p = os.path.join(folder, f"{pdb_id}_decoys_byte.pkl")
            if not os.path.exists(decoy_p):
                decoy_p = os.path.join(folder, f"{pdb_id}_decoys.pkl")

            # Dosyaları oku ve LMDB'ye 'bytes' olarak yaz
            try:
                with open(binder_p, 'rb') as f:
                    txn.put(f"{pdb_id}_binder".encode(), f.read())
                
                if os.path.exists(decoy_p):
                    with open(decoy_p, 'rb') as f:
                        txn.put(f"{pdb_id}_decoys".encode(), f.read())
            except Exception as e:
                print(f"Hata ({pdb_id}): {e}")

    env.close()
    print("✅ LMDB Kasası Hazır!")

# Kullanım:
create_lmdb('/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020', 'pdbbind_fast.lmdb')