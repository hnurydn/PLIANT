import os
import subprocess
import pandas as pd
import shutil
import re
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# ==========================================
#               AYARLAR
# ==========================================

# Dosya Yolları (Önceki adımlardan gelenler)
TRAIN_FASTA = os.path.join(PLANET_ROOT, "datasets/pdbbind_sequences.fasta")
TEST_FASTA = os.path.join(PLANET_ROOT, "datasets/test_sequences_all_variants.fasta")

# CSV Dosyaları
INPUT_TRAIN_CSV = os.path.join(PLANET_ROOT, "datasets/PDBbind_structure_map_pkd_prepared.json") 
# NOT: TrainSet.csv dosyanın asıl yerini buraya yazmalısın. 
# Genelde: .../datasets/pdbbind2020/index/TrainSet.csv olur.
# Eğer o yoksa JSON'dan filtreleyip yeni CSV üreteceğiz.
TRAIN_CSV_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/index/TrainSet.csv")

# Çıktılar
CDHIT_OUT = os.path.join(PLANET_ROOT, "datasets/train_overlaps")  # CD-HIT sonuç dosyası adı
FINAL_TRAIN_CSV = os.path.join(PLANET_ROOT, "datasets/TrainSet_Strict_Cleaned.csv")
LEAKAGE_REPORT = os.path.join(PLANET_ROOT, "datasets/leakage_report.txt")

'''# Parametreler
    """CD-HIT yüklü mü kontrol eder."""
    if shutil.which("cd-hit-2d") is None:
        print("HATA: 'cd-hit-2d' komutu bulunamadı!")
        print("Lütfen önce kurulumu yapın: 'conda install -c bioconda cd-hit'")
        return False
    return True'''

def parse_clstr_file(clstr_path):
    """
    .clstr dosyasını okur ve silinmesi gereken PDB ID'lerini bulur.
    CD-HIT-2D Mantığı:
    - Veritabanı (Test Seti) kümeyi oluşturur.
    - Sorgu (Train Seti) buna benzerse kümeye eklenir.
    - Train setinden gelen ve bir kümeye giren her şey SIZINTIDIR.
    """
    leaked_ids = set()
    
    with open(clstr_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Örnek satır: 1	148aa, >1fkb... at 95.00%
        # Bu satır, 1fkb'nin test setindeki bir şeye %95 benzediğini söyler.
        if "at " in line and "%" in line:
            # PDB ID'yi çek: > işaretinden sonraki ilk kelime
            match = re.search(r'>(\w+)', line)
            if match:
                pdb_id = match.group(1)
                # Test seti ID'leri 'LITPCBA_' veya 'DUDE_' ile başlıyor.
                # Train seti ID'leri ise sadece 4 karakterli (örn: 1fkb).
                # Biz sadece Train setinden olanları (kısa olanları) silmeliyiz.
                if not pdb_id.startswith("LITPCBA_") and not pdb_id.startswith("DUDE_"):
                    leaked_ids.add(pdb_id)
                    
    return leaked_ids

def main():
    if not check_cdhit_installed(): return

    print("--- CD-HIT İle Sızıntı Kontrolü Başlıyor ---")
    
    # 1. CD-HIT-2D Çalıştır
    # -i : Test Seti (Referans/Veritabanı)
    # -i2: Train Seti (Sorgu/Elenen)
    # -c : Benzerlik Eşiği (0.90)
    # -n : Word size (5, 0.90 için önerilen)
    # -d : Description length (0 = tam başlık)
    cmd = [
        "cd-hit-2d", 
        "-i", TEST_FASTA, 
        "-i2", TRAIN_FASTA, 
        "-o", CDHIT_OUT, 
        "-c", str(THRESHOLD), 
        "-n", "5",
        "-d", "0" 
    ]
    
    print(f"Komut çalıştırılıyor: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CD-HIT hatası: {e}")
        return

    # 2. Sonuçları Analiz Et
    clstr_file = CDHIT_OUT + ".clstr"
    print(f"Sonuç dosyası analiz ediliyor: {clstr_file}")
    
    leaked_pdbs = parse_clstr_file(clstr_file)
    print(f"⚠️ Tespit Edilen Sızıntı Sayısı: {len(leaked_pdbs)}")
    
    # 3. CSV Dosyasını Temizle ve Kaydet
    if os.path.exists(TRAIN_CSV_PATH):
        df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
        initial_count = len(df)
        
        # PDB kodlarını küçük harfe çevirip kontrol et
        # (FASTA'dan gelenler genelde küçük harftir ama garanti olsun)
        df['temp_id'] = df['PDB_code'].astype(str).str.lower()
        
        # Filtreleme: Yasaklı listede OLMAYANLARI tut
        clean_df = df[~df['temp_id'].isin(leaked_pdbs)].copy()
        clean_df.drop(columns=['temp_id'], inplace=True)
        
        final_count = len(clean_df)
        removed_count = initial_count - final_count
        
        # Kaydet
        clean_df.to_csv(FINAL_TRAIN_CSV)
        
        # Rapor Yaz
        with open(LEAKAGE_REPORT, "w") as f:
            f.write(f"Taranan Train Verisi: {initial_count}\n")
            f.write(f"Silinen (Leakage): {removed_count}\n")
            f.write(f"Kalan Temiz Veri: {final_count}\n")
            f.write("-" * 30 + "\n")
            f.write("Silinen PDB ID'leri:\n")
            for pid in leaked_pdbs:
                f.write(f"{pid}\n")
        
        print(f"\n✅ BAŞARILI!")
        print(f"Eski Veri Sayısı: {initial_count}")
        print(f"Silinen Veri: {removed_count}")
        print(f"Yeni Temiz Veri: {final_count}")
        print(f"Temiz dosya kaydedildi: {FINAL_TRAIN_CSV}")
        print(f"Silinenlerin listesi: {LEAKAGE_REPORT}")
        
    else:
        print(f"HATA: Train CSV dosyası bulunamadı: {TRAIN_CSV_PATH}")
        print("Lütfen script içindeki 'TRAIN_CSV_PATH' değişkenini doğru dosya yoluyla güncelleyin.")

if __name__ == "__main__":
    main()