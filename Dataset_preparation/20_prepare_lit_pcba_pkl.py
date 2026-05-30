import json
import os
import pickle
import sys
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import Pool, cpu_count
import traceback

# --- AYARLAR ---
# 1. Hazirladigin JSON dosyasi (DUZELTILDI: .pdb uzantili JSON kullaniliyor)
INPUT_JSON = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_pdb.json"
# 2. Cikti klasoru
OUTPUT_DIR = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA/lit_pcba_pkl_data"
# 3. Kutuphane yolu
CHEMUTILS_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation"

# Chemutils yolunu ekle
if CHEMUTILS_PATH not in sys.path:
    sys.path.append(CHEMUTILS_PATH)

try:
    # --- YINE YAMALI DOSYAYI KULLANIYORUZ (RDKit Mol destegi icin) ---
    from chemutils_dude import ComplexPocket
    print("✅ chemutils_dude (PATCHED) basariyla yuklendi.", flush=True)
except ImportError:
    print("❌ HATA: chemutils_dude.py bulunamadi! Onceki adimlardaki patch scriptini calistirin.", flush=True)
    sys.exit(1)

# RDKit uyarilarini kapat
RDLogger.DisableLog('rdApp.*')

def load_sdf(filename):
    """Normal SDF dosyasindan molekulleri okur (GZ degil)."""
    mols = []
    if not os.path.exists(filename):
        return mols
        
    try:
        # GZ degil, duz SDF okuyoruz (Cunku biz uretiik)
        suppl = Chem.SDMolSupplier(filename, sanitize=False)
        for mol in suppl:
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    mols.append(mol)
                except:
                    continue
    except Exception as e:
        print(f"⚠️ Okuma hatasi ({filename}): {e}", flush=True)
    return mols

def process_target(target_name, record, output_folder):
    """Bir target için Active ve Decoy PKL dosyalarını oluşturur."""
    
    protein_path = record['protein_file'] # JSON'da key ismini boyle yapmistik
    actives_path = record['actives_file']
    decoys_path = record['decoys_file']

    # 2. Actives Isle
    actives_pkl = os.path.join(output_folder, f"{target_name}_actives.pkl")
    if not os.path.exists(actives_pkl):
        active_mols = load_sdf(actives_path)
        active_pockets = []
        for mol in active_mols:
            try:
                # 3D koordinatlar zaten hizali oldugu icin ref_mol'e gerek yok
                pocket = ComplexPocket(protein_path, mol, pK=5.0) 
                active_pockets.append(pocket)
            except Exception as e:
                continue
        
        if active_pockets:
            with open(actives_pkl, 'wb') as f:
                pickle.dump(active_pockets, f)

    # 3. Decoys Isle
    decoys_pkl = os.path.join(output_folder, f"{target_name}_decoys.pkl")
    if not os.path.exists(decoys_pkl):
        decoy_mols = load_sdf(decoys_path)
        decoy_pockets = []
        for mol in decoy_mols:
            try:
                pocket = ComplexPocket(protein_path, mol, pK=0.0)
                decoy_pockets.append(pocket)
            except:
                continue
        
        if decoy_pockets:
            with open(decoys_pkl, 'wb') as f:
                pickle.dump(decoy_pockets, f)
    
    return True

# --- Wrapper Fonksiyonu ---
def process_wrapper(args):
    target_name, record, output_dir = args
    try:
        result = process_target(target_name, record, output_dir)
        if result is not True:
            return False
        return True
    except Exception as e:
        print(f"❌ KRİTİK HATA ({target_name}): {e}", flush=True)
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"📂 JSON Okunuyor: {INPUT_JSON}", flush=True)
    if not os.path.exists(INPUT_JSON):
        print("❌ JSON dosyasi bulunamadi! Lutfen onceki adimdaki JSON scriptini calistirin.")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    tasks = []
    for target, records in data.items():
        if not records: continue
        rec = records[0] # Her target icin ilk kaydi baz aliyoruz
        tasks.append((target, rec, OUTPUT_DIR))

    num_cores = 40 
        
    print(f"🚀 {len(tasks)} hedef, {num_cores} cekirdek ile isleniyor...", flush=True)

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_wrapper, tasks), total=len(tasks)))

    print("\n✅ Tum islemler bitti! PKL dosyalari hazir.", flush=True)

if __name__ == "__main__":
    main()
