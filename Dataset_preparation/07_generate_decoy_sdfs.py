# 07_generate_decoy_sdfs.py
import pandas as pd
import os
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import json # JSON yüklemek için eklenmiştir
import sys

# Global olarak yüklenecek JSON verisi (Fonksiyonlar arasında paylaşım için)
PK_DATA = {} 

# --- AYARLAR ---
# Girdi: BANANA Skorlama çıktılarının bulunduğu klasör (05. adımdan)
BANANA_SCORES_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/banana_output_scores_filtered/'
# JSON Haritası (Prepared PDB yollarını içerir)
PK_MAP_JSON = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
# Orijinal Adaylar (SMILES'ı çekmek için)
ORIGINAL_CANDIDATES_CSV = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/decoy_candidates_find_1000_suitable.csv"
# Bu script için ara CSV çıktısı
INTERMEDIATE_CSV = os.path.join(os.path.dirname(PK_MAP_JSON), 'all_filtered_decoys_for_sdf.csv')

DEFAULT_N_JOBS = 16
# --- BİTİŞ AYARLARI ---


def combine_all_filtered_decoys():
    """BANANA skor klasöründeki tüm *_filtered.csv dosyalarını birleştirir."""
    print(f"BANANA skor klasörü taranıyor: {BANANA_SCORES_DIR}")
    
    all_filtered_decoys = []
    
    # 1. Decoy CSV dosyalarını bul ve yükle
    for filename in os.listdir(BANANA_SCORES_DIR):
        if filename.endswith('_filtered.csv'):
            try:
                # Dosya yolu
                filepath = os.path.join(BANANA_SCORES_DIR, filename)
                df = pd.read_csv(filepath)
                
                # Sütunları standartlaştır ve sadece gerekenleri al
                if 'PDB_ID' in df.columns and 'CHEMBL_ID' in df.columns:
                    # Not: SMILES'ı yüklemek için orijinal aday CSV'sine (4. adımdan) ihtiyaç duyulur.
                    # Ancak bu adımda SMILES'ın zaten filtered CSV'de olduğunu varsayalım
                    # veya geçici olarak SMILES'ı CHEMBL_ID'den çekmek gerekir.
                    # Basit tutmak için, PDB_ID, CHEMBL_ID ve SMILES'ın bir şekilde birleştirildiğini varsayıyoruz.
                    # Ancak BANANA çıktısında SMILES yoktu. KRİTİK EKSİKLİK.

                    #!!! DÜZELTME: Bu noktada, skorlama aşamasından sadece CHEMBL_ID ve PDB_ID gelir.
                    # SMILES'ı bulmak için 4. adımdaki büyük aday CSV'sine ihtiyacımız var.
                    
                    # Şimdilik, filtrelenen ID'leri tutalım ve SMILES'ı sonraki birleştirme adımına bırakalım.
                    # VEYA basitçe: Tekrar birleştirme adımını atlayıp, SMILES'ı içeren birleştirilmiş bir CSV'nin
                    # manuel olarak hazırlandığını varsayalım (bu, orijinal pipeline'ınızdaki 'final_scored_chembl_smiles.csv'dir).
                    # Aşağıdaki kısım, o tek CSV'nin tüm PDB ID'lerini içerdiğini varsayar.

                    all_filtered_decoys.append(df)
            except Exception as e:
                print(f"UYARI: {filename} dosyası okunamadı: {e}")
                
    if not all_filtered_decoys:
        print("HATA: Filtrelenmiş hiçbir decoy CSV'si bulunamadı. Lütfen 05_pocket_create_and_score.py çıktısını kontrol edin.")
        return None

    # Tüm filtrelenmiş decoy'ları birleştir
    combined_df = pd.concat(all_filtered_decoys, ignore_index=True)
    
    # SMILES'ı içeren orijinal aday listesi (04_prepare_decoy_candidates.sh çıktısı)
    # SMILES'ı çekmek için bu dosyayı yüklemeliyiz.
    ORIGINAL_CANDIDATES_CSV = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/decoy_candidates_find_100_suitable.csv"
    try:
        df_original = pd.read_csv(ORIGINAL_CANDIDATES_CSV)
        df_original = df_original[['binder_pdb_id', 'chembl_id', 'chembl_smiles']].drop_duplicates()
        
        # Merge işlemi: PDB ID ve CHEMBL ID'ye göre SMILES'ı ekle
        combined_df = combined_df.merge(
            df_original, 
            left_on=['PDB_ID', 'CHEMBL_ID'], 
            right_on=['binder_pdb_id', 'chembl_id'], 
            how='left'
        )
        combined_df = combined_df[['PDB_ID', 'CHEMBL_ID', 'chembl_smiles']].drop_duplicates()
        combined_df.columns = ['PDB_ID', 'CHEMBL_ID', 'SMILES'] # Sütunları standartlaştır

        print(f"Toplam {len(combined_df)} filtrelenmiş Decoy (SMILES dahil) bulundu.")
        combined_df.to_csv(INTERMEDIATE_CSV, index=False)
        return combined_df
        
    except FileNotFoundError:
        print(f"KRİTİK HATA: Orijinal aday CSV'si ({ORIGINAL_CANDIDATES_CSV}) bulunamadı. SMILES çekilemiyor.")
        return None
    except Exception as e:
        print(f"KRİTİK HATA: SMILES birleştirme hatası: {e}")
        return None



# --- SDF OLUŞTURMA İŞLEVİ (JSON'dan yol çeker) ---

def generate_sdf_for_row_to_pdb_dir(row_data):
    """
    Tek bir filtrelenmiş decoy için 3D SDF oluşturur ve onu ilgili PDB klasörüne kaydeder.
    """
    pdb_id = row_data['PDB_ID']
    chembl_id = row_data['CHEMBL_ID']
    smiles = row_data['SMILES']
    
    # KRİTİK 1: JSON'da PDB ID'si var mı?
    if pdb_id not in PK_DATA:
        return (pdb_id, chembl_id, "HATA: PDB ID JSON haritasında yok.")

    # KRİTİK 2: JSON'dan protein yolunu al
    # ligand_sdf yolunu kullanmak, klasör yolunu bulmak için daha güvenlidir.
    reference_file_path = PK_DATA[pdb_id].get('ligand_sdf') 

    if not reference_file_path:
         return (pdb_id, chembl_id, "HATA: JSON'da Ligand yolu eksik.")

    # PDB klasörünün yolunu elde et
    # Örn: /.../2011-2019/1icj/1icj_ligand.sdf -> /.../2011-2019/1icj/
    target_dir = os.path.dirname(reference_file_path)
    output_path = os.path.join(target_dir, f"{chembl_id}.sdf")

    if not os.path.isdir(target_dir):
        return (pdb_id, chembl_id, "HATA: PDB klasörü (yolundan) bulunamadı.")
    
    # --- RESUMABILITY KONTROLÜ ---
    if os.path.exists(output_path):
        return (pdb_id, chembl_id, "Başarılı (Zaten Mevcut)")

    if not smiles or pd.isna(smiles):
        return (pdb_id, chembl_id, "HATA: SMILES boş (NaN)")

    try:
        # 3D Yapı Oluşturma
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (pdb_id, chembl_id, "HATA: RDKit SMILES parse edemedi")
        
        mol = Chem.AddHs(mol)
        
        embed_status = AllChem.EmbedMolecule(mol, randomSeed=42) 
        if embed_status == -1: 
            return (pdb_id, chembl_id, "HATA: RDKit EmbedMolecule başarısız")

        AllChem.UFFOptimizeMolecule(mol)
        
        # SDF Dosyası Olarak Kaydet
        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()
        
        return (pdb_id, chembl_id, "Başarılı")

    except Exception as e:
        return (pdb_id, chembl_id, f"KRİTİK HATA: {e}")

# --- ANA ÇALIŞTIRMA ---

def main():
    global PK_DATA
    
    parser = argparse.ArgumentParser(description="Adım 6: BANANA filtrelenmiş Decoy'ların 3D SDF'lerini ilgili PDB klasörlerine oluşturur.")
    parser.add_argument('-n', '--njobs', type=int, default=DEFAULT_N_JOBS,
                        help=f"Kullanılacak paralel işlem (CPU) sayısı. Varsayılan: {DEFAULT_N_JOBS}")
    args = parser.parse_args()

    # 1. JSON Haritasını Yükle (Global PK_DATA'yı doldurur)
    try:
        with open(PK_MAP_JSON, 'r') as f:
            PK_DATA = json.load(f)
        print(f"JSON Haritası Yüklendi. Toplam {len(PK_DATA)} PDB ID'si içeriyor.")
    except Exception as e:
        print(f"KRİTİK HATA: JSON haritası ({PK_MAP_JSON}) yüklenemedi: {e}")
        sys.exit(1)

    print("--- 7. Adım Başlatıldı: Filtrelenmiş Decoy SDF Oluşturma ---")
    
    # 2. Tüm filtrelenmiş decoy'ları SMILES ile birleştir
    # Not: combine_all_filtered_decoys fonksiyonu da global PK_DATA'ya erişebilir ancak
    # burada sadece CSV'ler ile işlem yaptığı için şimdilik global PK_DATA'yı kullanmasına gerek yok.
    df_combined = combine_all_filtered_decoys()
    if df_combined is None or df_combined.empty:
        print("İşlenecek filtrelenmiş decoy bulunamadı. Durduruluyor.")
        return
        
    # 3. İşleri Hazırla
    tasks = df_combined.to_dict('records')
    total_tasks = len(tasks)
        
    print(f"Toplam {total_tasks} decoy 3D SDF'sine dönüştürülecek ve PDB klasörlerine kaydedilecek.")

    # 4. Multiprocessing Pool'u Başlat
    
    results = []
    success_count = 0
    exist_count = 0
    fail_count = 0

    with Pool(processes=args.njobs) as pool:
        pbar = tqdm(pool.imap_unordered(generate_sdf_for_row_to_pdb_dir, tasks), 
                    total=total_tasks, 
                    desc="3D SDF'ler Oluşturuluyor ve Kopyalanıyor")
        
        for (pdb_id, chembl_id, status) in pbar:
            if status == "Başarılı":
                success_count += 1
            elif status == "Başarılı (Zaten Mevcut)":
                exist_count += 1
            else:
                fail_count += 1
                results.append((pdb_id, chembl_id, status)) 
                
    # 5. Özeti Yazdır
    print("\n--- İşlem Tamamlandı: Özet ---")
    print(f"✅ Yeni Oluşturulan SDF: {success_count}")
    print(f"➡️ Zaten Mevcut (Atlandı): {exist_count}")
    print(f"❌ Başarısız: {fail_count}")

    if fail_count > 0:
        print(f"\n--- Başarısız {fail_count} Decoy (İlk 10 Detay) ---")
        for i, (pdb_id, chembl_id, status) in enumerate(results):
            if i < 10:
                print(f"PDB: {pdb_id}, Decoy: {chembl_id} -> Durum: {status}")
        if fail_count > 10:
            print(f"(... ve {fail_count - 10} diğer hata ...)")

if __name__ == "__main__":
    main()

