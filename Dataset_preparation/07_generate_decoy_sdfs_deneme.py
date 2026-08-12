import pandas as pd
import os
import argparse
import numpy as np
from multiprocessing import Pool
import json
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- AYARLAR (Burayı kendi yollarına göre teyit et) ---
# --- AYARLAR (Burayı kendi yollarına göre teyit et) ---
BANANA_SCORES_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/banana_output_scores_filtered_3/")
PK_MAP_JSON = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json")
ORIGINAL_CANDIDATES_CSV = os.path.join(PLANET_ROOT, "datasets/decoy_candidates_all.csv")
FINAL_INDEX_CSV = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/decoy_generation_summary_index.csv")
DEFAULT_N_JOBS = 10 
# --- AYARLAR ---
# Yeni eklenecek satır:
DECOYS_OUTPUT_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/decoys_sdf_3")
os.makedirs(DECOYS_OUTPUT_DIR, exist_ok=True)
# Global değişken (Process'ler arası paylaşım için)
PK_DATA = {}

def combine_and_get_data():
    """Tüm CSV'leri birleştirir ve SMILES ekler. (Tip korumalı versiyon)"""
    print(f"BANANA skorları taranıyor: {BANANA_SCORES_DIR}")
    all_dfs = []
    
    # Banana skorlarını okurken PDB_ID'yi string olmaya zorluyoruz
    for filename in os.listdir(BANANA_SCORES_DIR):
        if filename.endswith('_filtered.csv'):
            file_path = os.path.join(BANANA_SCORES_DIR, filename)
            # DTYPE=STR EKLENDİ: Sayı gibi görünen ID'leri bozmaz
            df = pd.read_csv(file_path, dtype={'PDB_ID': str, 'CHEMBL_ID': str})
            all_dfs.append(df)
    
    if not all_dfs:
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("SMILES bilgileri orijinal aday listesinden çekiliyor...")
    try:
        # Orijinal listeyi okurken de string zorlaması yapıyoruz
        df_original = pd.read_csv(ORIGINAL_CANDIDATES_CSV, dtype={'binder_pdb_id': str, 'chembl_id': str})
        df_original = df_original[['binder_pdb_id', 'chembl_id', 'chembl_smiles']].drop_duplicates()
        
        # Merge işlemi (String vs String olduğu için artık güvenli)
        combined_df = combined_df.merge(
            df_original, 
            left_on=['PDB_ID', 'CHEMBL_ID'], 
            right_on=['binder_pdb_id', 'chembl_id'], 
            how='left'
        )
        
        final_df = combined_df[['PDB_ID', 'CHEMBL_ID', 'chembl_smiles']].drop_duplicates()
        final_df.columns = ['PDB_ID', 'CHEMBL_ID', 'SMILES']
        
        # Olası NaN değerleri temizle
        final_df = final_df.dropna(subset=['SMILES'])
        
        # PDB ID'lerdeki olası boşlukları veya .0 uzantılarını temizle (Garanti olsun)
        final_df['PDB_ID'] = final_df['PDB_ID'].apply(lambda x: str(x).replace('.0', '') if str(x).endswith('.0') else str(x))
        
        print(f"Merge Başarılı! {len(final_df)} adet aday işlenecek.")
        return final_df
        
    except Exception as e:
        print(f"SMILES birleştirme hatası: {e}")
        return None
    
    
def process_pdb_group(group_data):
    pdb_id, group = group_data
    
    # PDB ID'yi string'e çevir (Garanti olsun)
    pdb_id = str(pdb_id)
    
    # JSON kontrolü
    if pdb_id not in PK_DATA:
        # Eğer hala bulamıyorsa log'a yaz ama devam etme
        # (Artık sayı sorunu çözüldüğü için buraya çok düşmemeli)
        return []

    sdf_path = os.path.join(DECOYS_OUTPUT_DIR, f"{pdb_id}_decoys.sdf")
    
    # Dosya zaten var ve doluysa atla
    """if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
        return []"""

    records = []
    writer = Chem.SDWriter(sdf_path)
    count_written = 0
    
    for _, row in group.iterrows():
        chembl_id = row['CHEMBL_ID']
        smiles = row['SMILES']
        
        if not smiles: continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)
                
                # --- İYİLEŞTİRME: Daha esnek molekül üretimi ---
                # useRandomCoords=True: Zor molekülleri kıvırmak için rastgele koordinat dener
                # maxAttempts=50: Daha fazla dener
                res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, maxAttempts=50)
                
                if res != -1: # Başarılıysa
                    try:
                        AllChem.UFFOptimizeMolecule(mol)
                    except:
                        pass # Optimizasyon patlarsa en azından embed edilmiş halini kaydet
                    
                    mol.SetProp("_Name", str(chembl_id))
                    mol.SetProp("CHEMBL_ID", str(chembl_id))
                    mol.SetProp("PARENT_PDB", str(pdb_id))
                    
                    writer.write(mol)
                    records.append({'PDB_ID': pdb_id, 'CHEMBL_ID': chembl_id, 'SDF_PATH': sdf_path})
                    count_written += 1
        except:
            continue
            
    writer.close()
    
    # Dosya boşsa sil
    if count_written == 0 and os.path.exists(sdf_path):
        os.remove(sdf_path)
        return []

    return records


def main():
    global PK_DATA
    
    # 1. JSON Yükle
    with open(PK_MAP_JSON, 'r') as f:
        PK_DATA = json.load(f)

    # 2. Veriyi Hazırla
    df_combined = combine_and_get_data()
    if df_combined is None:
        print("İşlenecek veri bulunamadı!"); return

    # PDB_ID bazlı gruplandır
    groups = list(df_combined.groupby('PDB_ID'))
    
    # --- DÜZELTME 2: Zaten bitenleri listeden çıkar (Preprocessing) ---
    # Pool'a göndermeden önce zaten dosyası olanları listeden elersek
    # multiprocess başlatma maliyetinden (overhead) kurtuluruz. Çok hızlandırır.
    
    groups_to_process = []
    skipped_count = 0
    
    print("Mevcut dosyalar kontrol ediliyor...")
    for pdb_id, group in tqdm(groups, desc="Ön Kontrol"):
        sdf_path = os.path.join(DECOYS_OUTPUT_DIR, f"{pdb_id}_decoys.sdf")
        # Dosya var ve boş değilse listeye ekleme
        """if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            skipped_count += 1
        else:
            groups_to_process.append((pdb_id, group))"""
            
        groups_to_process.append((pdb_id, group))
            
    print(f"Toplam {len(groups)} PDB klasörü bulundu.")
    print(f"--> {skipped_count} tanesi zaten tamamlanmış (Atlanacak).")
    print(f"--> {len(groups_to_process)} tanesi işlenecek.")
    # ------------------------------------------------------------------

    if not groups_to_process:
        print("İşlenecek yeni PDB kalmadı.")
        return

    # 3. Multiprocessing ile İşle
    all_index_records = []
    with Pool(processes=DEFAULT_N_JOBS) as pool:
        # Sadece işlenmesi gerekenleri gönderiyoruz
        results = list(tqdm(pool.imap_unordered(process_pdb_group, groups_to_process), 
                           total=len(groups_to_process), 
                           desc="SDF Yazımı (Yeni PDB'ler)"))
        
        for sublist in results:
            all_index_records.extend(sublist)

    # 4. İndeksi Kaydet (APPEND MODU)
    if all_index_records:
        new_df = pd.DataFrame(all_index_records)
        
        # Eğer dosya yoksa başlık (header) koy, varsa koyma (mode='a')
        file_exists = os.path.isfile(FINAL_INDEX_CSV)
        
        new_df.to_csv(FINAL_INDEX_CSV, mode='a', index=False, header=not file_exists)
        
        print(f"\nİşlem bitti!")
        print(f"{len(new_df)} YENİ decoy indeksi dosyaya EKLENDİ (Append).")
    else:
        print("\nYeni decoy üretilmedi (Hata veya hepsi boş).")

    print(f"İndeks dosyası: {FINAL_INDEX_CSV}")

if __name__ == "__main__":
    main()
