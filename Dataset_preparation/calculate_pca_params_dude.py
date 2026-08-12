import h5py
import numpy as np
import pickle
import os
from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- KRİTİK AYARLAR ---
# 1. Eğitimden gelen PCA (Buldun!)
PCA_PKL_PATH = os.path.join(PLANET_ROOT, "esm2_pca_params.pkl") 

# 2. Ham DUD-E H5 (1280 boyutlu olan)
INPUT_H5 = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_esm2_embeddings.h5")

# 3. ÇIKTI: Modelin Yiyeceği Dosya (512 boyutlu olacak)
OUTPUT_H5 = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_esm2_embeddings_PCA512.h5")

def apply_pca_transform():
    print(f"\n🚀 PCA Dönüşümü Başlıyor...")
    print(f"🔑 Anahtar Dosya (PDBBind PCA): {PCA_PKL_PATH}")
    print(f"📄 Girdi Dosya (DUD-E Ham): {INPUT_H5}")
    
    if not os.path.exists(PCA_PKL_PATH):
        print("❌ HATA: PCA dosyası yok! Yolu kontrol et.")
        return

    # PCA Yükle
    with open(PCA_PKL_PATH, 'rb') as f:
        pca_data = pickle.load(f)
    print("✅ PCA parametreleri yüklendi.")

    # Parametreleri Ayrıştır
    mean_vec = None
    components = None
    
    # Eğer dosya sözlükse (dict)
    if isinstance(pca_data, dict):
        mean_vec = pca_data.get('mean')
        components = pca_data.get('components') # Shape: (512, 1280)
        print("ℹ️ Yapı: Sözlük (Dict)")
    # Eğer dosya sklearn objesi ise
    elif hasattr(pca_data, 'mean_') and hasattr(pca_data, 'components_'):
        mean_vec = pca_data.mean_
        components = pca_data.components_
        print("ℹ️ Yapı: Sklearn Objesi")
    else:
        print(f"❌ HATA: Bilinmeyen PCA formatı: {type(pca_data)}")
        return

    # Boyut Kontrolü
    print(f"   Mean Shape: {mean_vec.shape}")
    print(f"   Components Shape: {components.shape}") # (512, 1280) olmalı

    # H5 İşlemi
    with h5py.File(INPUT_H5, 'r') as h5_in, h5py.File(OUTPUT_H5, 'w') as h5_out:
        total_keys = len(h5_in.keys())
        print(f"🔄 {total_keys} protein dönüştürülüyor...")
        
        for target_name in tqdm(h5_in.keys()):
            # 1. Ham veriyi al (N, 1280)
            raw_data = h5_in[target_name][:]
            
            # 2. Matematiksel Dönüşüm (Manuel Transform)
            # Formül: (X - Mean) @ Components.T
            # X: (N, 1280)
            # Mean: (1280,)
            # Components.T: (1280, 512)
            
            # Merkezleme (Centering)
            data_centered = raw_data - mean_vec
            
            # Yansıtma (Projection)
            data_pca = np.dot(data_centered, components.T)
            
            # Veri tipini float32 yap (Model bunu sever)
            data_pca = data_pca.astype(np.float32)
            
            # 3. Kaydet (N, 512)
            h5_out.create_dataset(target_name, data=data_pca)

    print(f"\n✅ İŞLEM TAMAMLANDI!")
    print(f"💾 Yeni Dosya: {OUTPUT_H5}")
    print("👉 Bu dosyayı benchmark scriptine ver.")

if __name__ == "__main__":
    apply_pca_transform()