import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- AYARLAR ---
DATA_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_pkl_data")
CHEMUTILS_PATH = os.path.join(PLANET_ROOT, "Dataset_preparation")

if CHEMUTILS_PATH not in sys.path:
    sys.path.append(CHEMUTILS_PATH)

try:
    from chemutils_dude import ComplexPocket, Residue, Mol
    print("✅ chemutils_dude yüklendi.")
except ImportError:
    print("❌ chemutils_dude bulunamadı! Önceki adımlarda oluşturduğumuz dosyanın yerinde olduğundan emin ol.")
    sys.exit(1)

def fix_pocket_object(pocket):
    """
    Tek bir ComplexPocket nesnesini inceler ve alpha_coordinates eksikse tamamlar.
    """
    needs_update = False
    
    # Koordinat kontrolü: Yoksa veya boşsa
    if not hasattr(pocket, 'alpha_coordinates') or \
       (hasattr(pocket.alpha_coordinates, 'shape') and pocket.alpha_coordinates.shape[0] == 0) or \
       (isinstance(pocket.alpha_coordinates, list) and len(pocket.alpha_coordinates) == 0):
        
        needs_update = True
        new_coords = []
        
        for residue in pocket.pocket_residues:
            # Residue içinden koordinat almaya çalış
            if hasattr(residue, 'get_alpha_position'):
                coords = residue.get_alpha_position()
            elif hasattr(residue, 'alpha_position'):
                coords = residue.alpha_position
            else:
                coords = None
            
            # Eğer koordinat yoksa veya boşsa -> Mass Center kullan
            if coords is None or len(coords) == 0 or (hasattr(coords, 'shape') and coords.shape[0] == 0):
                if hasattr(residue, 'get_mass_center'):
                    mc = residue.get_mass_center()
                else:
                    mc = residue.mass_center
                
                if hasattr(mc, 'reshape'):
                    mc = mc.reshape(1, 3)
                new_coords.append(mc)
            else:
                if hasattr(coords, 'reshape'):
                    coords = coords.reshape(1, 3)
                new_coords.append(coords)
        
        # Listeyi birleştirip Tensor yap ve nesneye kaydet
        if len(new_coords) > 0:
            coords_np = np.concatenate(new_coords, axis=0).astype(np.float32)
            pocket.alpha_coordinates = torch.from_numpy(coords_np)
        else:
            # Hiçbir şey bulunamadıysa (olmamalı ama güvenlik)
            pocket.alpha_coordinates = torch.zeros((0, 3))
            
    return needs_update, pocket

def main():
    # Klasördeki tüm PKL dosyalarını bul
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
    print(f"🚀 Toplam {len(all_files)} dosya kalıcı olarak onarılacak...")

    fixed_count = 0
    
    for filename in tqdm(all_files, desc="Onarılıyor"):
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            # 1. Dosyayı Oku
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                continue
                
            # 2. İçindeki Nesneleri Düzelt
            file_modified = False
            fixed_data = []
            
            for pocket in data:
                is_fixed, new_pocket = fix_pocket_object(pocket)
                if is_fixed:
                    file_modified = True
                fixed_data.append(new_pocket)
            
            # 3. Eğer değişiklik yapıldıysa üzerine kaydet
            if file_modified:
                # Önce geçici dosyaya yaz (Güvenlik için)
                temp_path = file_path + ".tmp"
                with open(temp_path, 'wb') as f:
                    pickle.dump(fixed_data, f)
                
                # Başarılı olursa orijinalin üzerine taşı
                os.replace(temp_path, file_path)
                fixed_count += 1
                
        except Exception as e:
            print(f"\n❌ HATA ({filename}): {e}")
            continue

    print(f"\n✅ İŞLEM TAMAMLANDI!")
    print(f"Toplam {fixed_count} dosya güncellendi ve onarıldı.")
    print("Artık benchmark kodunu herhangi bir yama olmadan çalıştırabilirsin.")

if __name__ == "__main__":
    main()