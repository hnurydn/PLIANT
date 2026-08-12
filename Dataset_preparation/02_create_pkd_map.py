# 02_create_pkd_map.py
import os
import json
import glob
import math
import re
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- Yapılandırma ---
# İŞİN ÇALIŞTIĞI ANA DİZİN (Scratch)
BASE_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020")
# Yeni JSON dosyasının adı
OUTPUT_JSON = os.path.join(BASE_DIR, "PDBbind_structure_map_pkd_prepared.json")
# PDBbind index dosyasının yolu (index klasöründe olduğunu varsayıyoruz)
INDEX_FILE = os.path.join(BASE_DIR, "index", "INDEX_general_PL.2020R1.lst")

# Taranacak ana yıl klasörleri
YEAR_FOLDERS = ["1981-2000", "2001-2010", "2011-2019"]

# --- Yeni Fonksiyonlar ---

def convert_to_pkd(affinity_string):
    """
    IC50, Kd, Ki değerlerini pK değerine (pK = -log10(Molar)) çevirir.
    """
    
    # Düzenli İfadeyi Genişletme:
    # 1. Afinite Tipi: (IC50|Kd|Ki)
    # 2. Operatör: ([<>=~]*) (şimdi ~ işaretini de içeriyor)
    # 3. Değer: ([0-9.]+)
    # 4. Birim: ([a-zA-Z]+)
    match = re.match(r"(IC50|Kd|Ki)([<>=~]*)([0-9.]+)([a-zA-Z]+)", affinity_string.strip(), re.IGNORECASE)
    
    if not match:
        # print(f"UYARI: Tanınmayan afinite formatı: {affinity_string}. Atlanıyor.")
        return None

    # Değerleri çıkar
    aff_type, operator, value_str, unit = match.groups()
    unit = unit.strip() # Birimdeki boşlukları temizle
    
    try:
        value = float(value_str)
    except ValueError:
        return None

    # Birim dönüşümleri: Molar (M) cinsine çevir
    unit_multipliers = {
        "M": 1.0,
        "mM": 1e-3,  # YENİ: Millimolar (10^-3)
        "uM": 1e-6,  # mikromolar
        "nM": 1e-9,  # nanomolar
        "pM": 1e-12, # pikomolar
    }
    
    if unit not in unit_multipliers:
        # Tekrar mM uyarısı almamak için burayı yoruma alabilirsiniz, 
        # ancak başka birimler gelirse uyarı vermek faydalıdır.
        # print(f"UYARI: Desteklenmeyen birim: {unit}. Atlanıyor.")
        return None

    # Konsantrasyonu Molar (M) cinsinden hesapla
    molar_concentration = value * unit_multipliers[unit]
    
    # pK değerini hesapla
    if molar_concentration > 0:
        pkd = -math.log10(molar_concentration)
        return pkd
    else:
        return None # Geçersiz konsantrasyon


def load_pdbbind_index(index_path):
    """
    PDBbind index dosyasını okur ve PDB ID'ye göre pKd değerlerini içeren bir sözlük döndürür.
    """
    pdb_affinity_map = {}
    print(f"PDBbind index dosyası yükleniyor: {index_path}")
    
    if not os.path.exists(index_path):
        print(f"HATA: Index dosyası bulunamadı: {index_path}")
        return {}

    with open(index_path, 'r') as f:
        for line in f:
            # Yorum satırlarını ve başlıkları atla
            if line.startswith('#'):
                continue
            
            # Satırı boşluklara göre ayır
            parts = line.split()
            
            # PDB kodu, çözünürlük, yıl, afinite...
            if len(parts) >= 4:
                pdb_id = parts[0].lower() # PDB kodlarını küçük harfe çevir
                affinity_str = parts[3]   # 4. sütun afinite değerini içerir
                
                pkd_value = convert_to_pkd(affinity_str)
                
                if pkd_value is not None:
                    # pkd_value = round(pkd_value, 4) # İsteğe bağlı olarak yuvarlayabilirsiniz
                    pdb_affinity_map[pdb_id] = pkd_value
                
    print(f"Index'ten {len(pdb_affinity_map)} adet pKd değeri yüklendi.")
    return pdb_affinity_map

# --- Ana Fonksiyon ---
def create_structure_map(base_dir, year_folders, index_file, output_json):
    """
    Yıl klasörlerini tarar ve dosya yollarını PDBbind afinite değerleriyle
    (pKd olarak) birlikte bir JSON haritasına kaydeder.
    """
    # 1. Adım: Index'ten pKd değerlerini yükle
    pdb_affinity_map = load_pdbbind_index(index_file)
    if not pdb_affinity_map:
        print("HATA: pKd değerleri yüklenemedi. İşlem durduruluyor.")
        return {}

    # 2. Adım: Yapısal dosyaları tara ve haritayı oluştur
    structure_map = {}
    found_complexes = 0
    
    print(f"Dosyalar {base_dir} altında taranıyor...")
    
    for year_folder in year_folders:
        year_path = os.path.join(base_dir, year_folder)
        
        if not os.path.isdir(year_path):
            print(f"UYARI: {year_path} klasörü bulunamadı veya dizin değil. Atlanıyor.")
            continue
            
        pdb_id_folders = glob.glob(os.path.join(year_path, "*/"))
        
        for folder_path in pdb_id_folders:
            pdb_id = os.path.basename(os.path.normpath(folder_path)).lower() # ID'yi küçük harfe çevir
            
            # pKd ve dosya varlığını kontrol et
            has_pkd = pdb_id in pdb_affinity_map
            
            protein_pdb = os.path.join(folder_path, f"{pdb_id}_protein_prepared.pdb")
            ligand_sdf = os.path.join(folder_path, f"{pdb_id}_ligand.sdf")

            # Gerekli dosyaların ve pKd'nin varlığını kontrol et
            if os.path.exists(protein_pdb) and os.path.exists(ligand_sdf) and has_pkd:
                structure_map[pdb_id] = {
                    "pK_d": pdb_affinity_map[pdb_id],  # pKd değerini ekle
                    "ligand_sdf": ligand_sdf,
                    "protein_pdb": protein_pdb
                }
                found_complexes += 1
                
    # 3. Adım: Haritayı JSON dosyasına kaydet
    print(f"Toplam {found_complexes} kompleks (pKd dahil) JSON dosyasına yazılıyor: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(structure_map, f, indent=4)
        
    print("İşlem tamamlandı. Yapısal harita pKd ile oluşturuldu.")
    return structure_map

# --- Çalıştırma ---
if __name__ == '__main__':
    create_structure_map(BASE_DIR, YEAR_FOLDERS, INDEX_FILE, OUTPUT_JSON)