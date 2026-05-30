import os, argparse, pickle, random, json
import sys
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Tuple

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.warning')
    # Chemutils kütüphanenin yüklü olduğundan emin ol
    from chemutils import ComplexPocket, Mol 
except ImportError as e:
    print(f"KRİTİK HATA: Kütüphaneler eksik: {e}", file=sys.stderr)
    sys.exit(1)

# --- AYARLAR (Sabit Yollar) ---
PK_MAP_PATH_LOCAL = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
# Yeni 1.6 GB'lık dosyadan üretilen SDF'lerin klasörü
DECOYS_SDF_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/decoys_sdf'
DEFAULT_PK = 0.0

def get_all_pdb_dirs(base_dir: str) -> List[str]:
    """base_dir altındaki PDB ID klasörlerini (4 karakterli) toplar."""
    all_records = []
    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_path):
            for pdb_dir in os.listdir(full_path):
                pdb_path = os.path.join(full_path, pdb_dir)
                # Sadece 4 karakterli klasörleri PDB ID kabul et
                if os.path.isdir(pdb_path) and len(pdb_dir) == 4:
                    all_records.append(pdb_path)
    return all_records

def process_PDBBind_Complex(record_dir: str, pK_data: dict) -> str:
    """
    1. Mevcut binder.pkl varsa atlar (aktif ligandı korur).
    2. decoys_sdf klasöründe SDF varsa ve _decoys.pkl yoksa oluşturur.
    """
    pdb_name = os.path.basename(record_dir)
    try:
        record = pK_data[pdb_name]
        pK_binder = record.get('pK_d', 0.0)
        protein_pdb = record['protein_pdb']
        ligand_sdf = record['ligand_sdf'] 
    except KeyError:
        return f"UYARI: {pdb_name} JSON haritasında yok."

    binder_pkl_path = os.path.join(record_dir, f"{pdb_name}_binder.pkl")
    decoys_pkl_path = os.path.join(record_dir, f"{pdb_name}_decoys.pkl")
    
    binder_pocket = None
    
    # ---------------------------------------------------------
    # 1. BINDER PKL İŞLEME (Aktif Ligand)
    # ---------------------------------------------------------
    try:
        if not os.path.exists(binder_pkl_path):
            if not os.path.exists(protein_pdb) or not os.path.exists(ligand_sdf):
                 return f"❌ HATA: Dosyalar eksik ({pdb_name})."
            
            # Yeni nesne oluştur
            binder_pocket = ComplexPocket(protein_pdb, ligand_sdf, pK_binder, decoy_sdf="")
            with open(binder_pkl_path, 'wb') as f:
                pickle.dump(binder_pocket, f, pickle.HIGHEST_PROTOCOL)
        else:
            # Varsa sadece içeriği oku (Decoy oluşturmak için şablon olarak kullanacağız)
            with open(binder_pkl_path, 'rb') as f:
                binder_pocket = pickle.load(f)
    except Exception as e:
        return f"❌ HATA: {pdb_name} Binder PKL - {e}"

    # ---------------------------------------------------------
    # 2. DECOY PKL OLUŞTURMA (Eksikleri Tamamla)
    # ---------------------------------------------------------
    if os.path.exists(decoys_pkl_path):
        return f"✅ ATLANDI: {pdb_name} decoy zaten var."

    decoy_sdf_source = os.path.join(DECOYS_SDF_DIR, f"{pdb_name}_decoys.sdf")

    if not os.path.exists(decoy_sdf_source):
        return f"⚠️ UYARI: {pdb_name} için SDF yok (Atlandı)."

    try:
        suppl = Chem.SDMolSupplier(decoy_sdf_source, removeHs=False)
        all_decoy_pockets = []
        
        # Hız için binder şablonunu bir kez serileştir
        binder_blob = pickle.dumps(binder_pocket)

        for mol in suppl:
            if mol is None: continue
            
            try:
                # Şablondan yeni bir kopya çıkar
                decoy_pocket = pickle.loads(binder_blob)
                decoy_pocket.ligand = Mol(mol) 
                decoy_pocket.pK = 0.0 
                decoy_pocket.decoys = [] 
                
                # Etkileşimi yeni liganda göre güncelle
                decoy_pocket.pro_lig_interaction = decoy_pocket.get_interaction_label()
                all_decoy_pockets.append(decoy_pocket)
            except:
                continue
        
        if all_decoy_pockets:
            with open(decoys_pkl_path, 'wb') as f:
                pickle.dump(all_decoy_pockets, f, pickle.HIGHEST_PROTOCOL)
            return f"✅ İŞLENDİ: {pdb_name} ({len(all_decoy_pockets)} decoy)."
        else:
            return f"⚠️ UYARI: {pdb_name} geçerli decoy üretilemedi."

    except Exception as e:
        return f"❌ HATA: {pdb_name} decoy işleme - {e}"

def collect_all_pkl_paths(pdb_dirs: List[str]) -> List[Tuple[str, float, bool]]:
    """Final eğitim listesini (all_records_for_training.pkl) oluşturur."""
    all_training_records = []
    with open(PK_MAP_PATH_LOCAL, 'r') as f:
        pK_data = json.load(f)

    for record_dir in tqdm(pdb_dirs, desc="Liste Hazırlanıyor"):
        pdb_name = os.path.basename(record_dir)
        
        # Binder Ekle (is_multi=False)
        binder_path = os.path.join(record_dir, f"{pdb_name}_binder.pkl")
        if os.path.exists(binder_path):
            pK = pK_data.get(pdb_name, {}).get('pK_d', DEFAULT_PK)
            all_training_records.append((binder_path, pK, False))
            
        # Decoy Paketini Ekle (is_multi=True)
        decoys_path = os.path.join(record_dir, f"{pdb_name}_decoys.pkl")
        if os.path.exists(decoys_path):
            all_training_records.append((decoys_path, DEFAULT_PK, True))
            
    return all_training_records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True, help="PDB ana dizini.")
    parser.add_argument('-n', '--njobs', required=True, type=int, help="CPU sayısı.")
    parser.add_argument('-o', '--output_path', default="all_records_for_training.pkl")
    args = parser.parse_args()
    
    with open(PK_MAP_PATH_LOCAL, 'r') as f:
        pK_data = json.load(f)
        
    pdb_dirs = get_all_pdb_dirs(args.dir)
    print(f"Toplam {len(pdb_dirs)} klasör taranıyor...")

    # PKL Oluşturma Havuzu
    with Pool(args.njobs) as pool:
        worker = partial(process_PDBBind_Complex, pK_data=pK_data)
        results = list(tqdm(pool.imap_unordered(worker, pdb_dirs), total=len(pdb_dirs), desc="PKL Üretimi"))
        
    # Eğitim listesini kaydet
    all_training_records = collect_all_pkl_paths(pdb_dirs)
    with open(args.output_path, 'wb') as out_f:
        pickle.dump(all_training_records, out_f, pickle.HIGHEST_PROTOCOL)
        
    print(f"✅ Başarılı! Toplam Kayıt: {len(all_training_records)}")

if __name__ == '__main__':
    main()