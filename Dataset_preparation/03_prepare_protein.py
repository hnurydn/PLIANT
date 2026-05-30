# 03_prepare_protein.py
import pdbfixer
from simtk.openmm.app import PDBFile
import os
import json
import argparse
from multiprocessing import Pool 
import sys
# tqdm modülü kafa karışıklığını önlemek için kaldırıldı, manuel loglama kullanılacak.

# --- Fonksiyon 1: Tek Bir PDB Dosyasını Hazırlama ---
def prepare_protein_for_planet(pdb_file_path):
    output_dir = os.path.dirname(pdb_file_path)

    # Çıktı yolunu hesapla
    base_name = os.path.basename(pdb_file_path).replace(".pdb", "_prepared.pdb")
    output_path = os.path.join(output_dir, base_name)
    
    # KRİTİK: ZATEN VAR OLANI ATLA (RESUMABILITY)
    if os.path.exists(output_path):
        # Başarılı sayılır ve JSON'a eklenir, ancak işleme gerek yok.
        print(f"[ATLANDI] {os.path.basename(pdb_file_path)}")
        sys.stdout.flush() # Anında logla
        return (pdb_file_path, output_path, "Already Exists") 

    try:
        # PDBFixer İşlemleri
        fixer = pdbfixer.PDBFixer(filename=pdb_file_path)
        fixer.findMissingResidues() 
        fixer.findMissingAtoms() 
        fixer.removeHeterogens(False) 
        fixer.addMissingAtoms() 
        fixer.addMissingHydrogens(7.0) 
        
        # Dosyayı kaydet
        with open(output_path, 'w') as outfile:
            PDBFile.writeFile(fixer.topology, fixer.positions, outfile)

        # Başarılı loglaması (İlerleme Takibi için)
        print(f"[BAŞARILI] {os.path.basename(pdb_file_path)}")
        sys.stdout.flush() # Logu hemen yazdır
        return (pdb_file_path, output_path, None) 

    except Exception as e:
        print(f"[HATA] {os.path.basename(pdb_file_path)} işlenemedi. Hata: {e}")
        sys.stderr.flush() # Hata logunu hemen yazdır
        return (pdb_file_path, None, str(e))

# --- Fonksiyon 2: JSON Haritasını İşleme ve Güncelleme ---
def process_map_for_proteins(json_path, njobs=1):
    print(f"JSON Haritası Yükleniyor: {json_path}")
    
    # 1. JSON YÜKLEME
    if not os.path.exists(json_path):
        print(f"KRİTİK HATA: JSON dosyası bulunamadı: {json_path}")
        return
    try:
        with open(json_path, 'r') as f:
            all_records_data = json.load(f)
    except Exception as e:
        print(f"KRİTİK HATA: JSON dosyası okunamadı veya bozuk: {e}")
        return
    
    if not all_records_data:
        print("UYARI: JSON haritası boş görünüyor. Hiçbir protein işlenmeyecek.")
        return

    # 2. İŞLENECEK VE ATLANACAK YOLLARI BELİRLE (Resumability Kontrolü)
    data_to_process = all_records_data.copy()
    paths_to_process = []
    initial_results = []
    all_original_paths = [v['protein_pdb'] for v in data_to_process.values()]
    
    print("Diskteki mevcut 'prepared' dosyalar kontrol ediliyor...")
    
    for original_path in all_original_paths:
        output_dir = os.path.dirname(original_path)
        base_name = os.path.basename(original_path).replace(".pdb", "_prepared.pdb")
        prepared_path = os.path.join(output_dir, base_name)
        
        if os.path.exists(prepared_path):
            initial_results.append((original_path, prepared_path, "Already Exists"))
        else:
            paths_to_process.append(original_path)
    
    total_initial = len(all_original_paths)
    
    print(f"Başlangıçta {len(initial_results)} protein bulundu (Atlanıyor). {len(paths_to_process)} adet işlenecek.")
    
    if not paths_to_process:
        results = initial_results
    
    else:
        # 3. KALAN İŞLERİ PARALEL ÇALIŞTIR (En sade pool.map)
        print(f"Kalan {len(paths_to_process)} protein {njobs} çekirdek ile işleniyor...")
        
        pool = Pool(njobs)
        
        # tqdm/chunking olmadan düz pool.map kullanılıyor
        new_results = pool.map(prepare_protein_for_planet, paths_to_process)
        
        pool.close()
        pool.join()
        
        # Yeni ve eski sonuçları birleştir
        results = initial_results + new_results

    # 4. JSON Haritasını Nihai Olarak Güncelle ve Kaydet
    data_to_update = all_records_data.copy()
    success_count = 0
    
    for original_path, prepared_path, error in results:
        if error is None or error == "Already Exists":
            success_count += 1
            pdb_id_full = os.path.basename(original_path).replace('_protein.pdb', '')
            
            if pdb_id_full in data_to_update:
                data_to_update[pdb_id_full]['protein_prepared_pdb'] = prepared_path 

    error_count = len(results) - success_count
    
    print("\n--- İŞLEM ÖZETİ ---")
    print(f"Toplam Başarılı Protein Hazırlığı Sayısı: {success_count} / {total_initial}")
    print(f"Hata Sayısı: {error_count}")
    print("--------------------")
    
    # 5. Yeni JSON dosyasını oluştur ve kaydet
    base, ext = os.path.splitext(json_path)
    new_json_path = base + "_prepared" + ext
    
    print(f"Güncellenmiş harita kaydediliyor: {new_json_path}")
    with open(new_json_path, 'w') as f:
        json.dump(data_to_update, f, indent=4)
        
    print(f"Yeni JSON yolu, sonraki adımlar için kritik: {new_json_path}")
    
# --- ANA ÇALIŞTIRMA BLOĞU (ARGPARSE) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDBbind proteinlerini PDBFixer ile hazırlar.")
    
    parser.add_argument('-j', '--json_path', required=True, type=str, 
                         help='PDBbind kompleks haritasını içeren JSON dosyasının tam yolu.')
    
    parser.add_argument('-n', '--njobs', required=True, type=int, 
                         help='Paralel olarak kullanılacak CPU çekirdeği sayısı.')
    
    args = parser.parse_args()
    
    process_map_for_proteins(args.json_path, njobs=args.njobs)
    
    print("Tüm protein dosyaları PDBFixer ile işlendi.")