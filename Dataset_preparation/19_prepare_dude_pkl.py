import json
import os
import pickle
import gzip
import sys
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import Pool, cpu_count
import traceback

# --- AYARLAR ---
INPUT_JSON = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/dude_paths_prepared_arf.json"
OUTPUT_DIR = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/dude_pkl_data"
CHEMUTILS_PATH = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation"

# Chemutils yolunu ekle
if CHEMUTILS_PATH not in sys.path:
    sys.path.append(CHEMUTILS_PATH)

try:
    # --- YAMALI DOSYAYI ÇAĞIRIYORUZ ---
    from chemutils_dude import ComplexPocket
    print("✅ chemutils_dude (PATCHED) basariyla yuklendi.", flush=True)
except ImportError:
    print("❌ HATA: chemutils_dude.py bulunamadi! Lutfen 'create_patched_chemutils.py' scriptini calistirin.", flush=True)
    sys.exit(1)

# RDKit uyarilarini kapat
RDLogger.DisableLog('rdApp.*')

def load_sdf_gz(filename):
    """GZ sıkıştırılmış SDF dosyasından molekülleri okur."""
    mols = []
    try:
        with gzip.open(filename, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f, sanitize=False)
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
    
    protein_path = record['protein_pdb']
    actives_path = record['actives_file']
    decoys_path = record['decoys_file']

    # 2. Actives Isle
    actives_pkl = os.path.join(output_folder, f"{target_name}_actives.pkl")
    if not os.path.exists(actives_pkl):
        active_mols = load_sdf_gz(actives_path)
        active_pockets = []
        for mol in active_mols:
            try:
                # DUD-E için ref_mol YOK, pK=5.0
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
        decoy_mols = load_sdf_gz(decoys_path)
        decoy_pockets = []
        for mol in decoy_mols:
            try:
                # DUD-E için ref_mol YOK, pK=0.0
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
            print(f"⚠️ Uyari ({target_name}): {result}", flush=True)
            return False
        return True
    except Exception as e:
        print(f"❌ KRITIK HATA ({target_name}): {e}", flush=True)
        # traceback.print_exc() 
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"📂 JSON Okunuyor: {INPUT_JSON}", flush=True)
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    tasks = []
    for target, records in data.items():
        if not records: continue
        rec = records[0]
        tasks.append((target, rec, OUTPUT_DIR))

    # Manuel olarak 40 cekirdek
    num_cores = 40 
        
    print(f"🚀 {len(tasks)} hedef, {num_cores} cekirdek ile isleniyor...", flush=True)

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_wrapper, tasks), total=len(tasks)))

    print("\n✅ Tum islemler bitti! PKL dosyalari hazir.", flush=True)

if __name__ == "__main__":
    main()
