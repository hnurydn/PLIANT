# calculate_pca_params.py

import h5py
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
from tqdm import tqdm

# --- PARAMETRELER ---
HDF5_YOLU = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020_esm2_embeddings_v2.h5' # Lütfen HDF5 dosyanızın adını buraya girin
CIKTI_YOLU = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/esm2_pca_params_dude.pkl'
HEDEF_BOYUT = 512 # 1280D'den 512D'ye düşürme, GNN için standart ve güvenli bir boyut

def calculate_pca_whitening_params(h5_path, output_path, target_dim):
    """
    Tüm ESM-2 gömülmeleri üzerinden PCA/Whitening parametrelerini hesaplar.
    """
    all_embeddings = []
    
    print("1. HDF5 dosyasındaki tüm ESM-2 gömülmeleri okunuyor...")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Tüm PDB ID'lerini al
            pdb_ids = list(f.keys())
            
            for pdb_id in tqdm(pdb_ids, desc="Gömülmeleri oku"):
                # Gömülmeleri oku (boyut: [residue_count, 1280])
                embedding = f[pdb_id][()] 
                
                # Tüm rezidü gömülmelerini tek bir listeye ekle
                all_embeddings.append(embedding)
                
    except Exception as e:
        print(f"Hata oluştu: HDF5 okunamadı: {e}")
        return

    # Tüm gömülmeleri tek bir NumPy matrisinde birleştir
    all_embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    print(f"\nToplam {all_embeddings_matrix.shape[0]} adet rezidü gömülmesi toplandı (Boyut: 1280D).")
    
    # 2. PCA Modelini Hesapla
    # whiten=True: Bu, PCA'nın beyazlatma özelliğini etkinleştirir.
    # Bu, veriyi ortalar (mean=0) ve varyansını birimler (variance=1).
    pca = PCA(n_components=target_dim, whiten=True) 
    
    print(f"2. PCA modelini ve Whitening matrisini hesaplanıyor...")
    
    # Tüm veriyi eğitiyoruz. Bu, tüm matrisi bellekte tutmayı gerektirir.
    pca.fit(all_embeddings_matrix)

    # 3. Parametreleri Kaydet
    pca_params = {
        'mean': pca.mean_.astype(np.float32),        # Ortalama vektör (1280D)
        'components': pca.components_.astype(np.float32), # Dönüşüm matrisi W (512x1280)
        'target_dim': target_dim
    }
    
    with open(output_path, 'wb') as pkl_file:
        pickle.dump(pca_params, pkl_file)
        
    print(f"✅ PCA/Whitening parametreleri başarıyla kaydedildi: {output_path}")
    print(f"Yeni gömülme boyutu: {target_dim}D")

if __name__ == '__main__':
    calculate_pca_whitening_params(HDF5_YOLU, CIKTI_YOLU, HEDEF_BOYUT)