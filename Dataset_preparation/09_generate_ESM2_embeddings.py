# 09_generate_ESM2_embeddings.py
import os
import json
import argparse
import time
import numpy as np
import torch
import esm
import h5py
from tqdm import tqdm
import warnings
from typing import List, Tuple

# HDF5'in string uyarılarını ve PyTorch'un non-persistent uyarılarını bastır
# warnings.filterwarnings("ignore", category=h5py.h5t.UnrecognizedTypeWarning)
# warnings.filterwarnings("ignore", "The given NumPy array is not writeable", UserWarning)

# --- AYARLAR ---
# PDBbind harita dosyanızın yolunu güncelleyin
DEFAULT_INPUT_JSON = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json"
OUTPUT_H5_FILE = 'pdbbind2020_esm2_embeddings_v2.h5'
PDB_ID_LIST_FILE = 'pdbbind2020_pdb_ids.txt' # İşlenen PDB ID'lerini tutmak için

# ESM-2 650M Model Parametreleri
MODEL_NAME = "esm2_t33_650M_UR50D"
REPR_LAYER = 33 
EMBEDDING_DIM = 1280 
# Sunucunuzun VRAM'ine göre ayarlayın. 1-8 arası deneme yapın.
BATCH_SIZE = 2

# 3 harfli amino asit kodunu 1 harfli koda çeviren sözlük
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}
# -----------------

# --- parse_fasta_from_pdb (GÜNCELLENDİ) ---
def parse_fasta_from_pdb(pdb_file_path: str) -> tuple:
    """
    PDB dosyasından amino asit dizisini ve (Chain, ResNum) -> Index eşleşmesini çıkarır.
    
    Döndürdüğü:
        full_sequence (str): Birleştirilmiş protein dizisi.
        mapping (dict): {(chain_id, res_num): 0_based_index} sözlüğü.
    """
    sequences = {}      # Zincir bazlı diziler: {'A': ['M', 'K', ...], 'B': ...}
    chain_res_info = {} # Zincir bazlı residue anahtarları: {'A': [('A', 1), ('A', 2)...]}
    chain_order = []    # Zincirlerin dosyada görülme sırası (Örn: ['A', 'B'])

    current_res_num = None # Tekrar eden atom satırlarını engellemek için

    try:
        with open(pdb_file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    # PDB Sütunları
                    chain_id = line[21].strip() or 'A'
                    res_num_str = line[22:26].strip()
                    res_type_3 = line[17:20].strip()

                    # Geçersiz veya bilinmeyen amino asitleri atla
                    if not res_num_str or res_type_3 not in AA_3_TO_1:
                        continue
                    
                    # O anki residue'nun kimliği
                    res_id_key = (chain_id, int(res_num_str))

                    # Yeni bir zincirle karşılaştıysak listeleri başlat
                    if chain_id not in sequences:
                        sequences[chain_id] = []
                        chain_res_info[chain_id] = []
                        chain_order.append(chain_id) # Sırayı kaydet

                    # Aynı residue içindeki atomları (CA, CB, vb.) tekrar tekrar eklememek için kontrol
                    if res_id_key != current_res_num:
                        one_letter = AA_3_TO_1.get(res_type_3, '')
                        
                        # Eğer geçerli bir harfse (X değilse) ekle
                        if one_letter:
                            # 1. Harfi diziye ekle
                            sequences[chain_id].append(one_letter)
                            # 2. Bu harfin kimliğini (Chain, Num) kaydet
                            chain_res_info[chain_id].append(res_id_key)
                            
                            current_res_num = res_id_key
                            
    except Exception as e:
        print(f"PDB okuma hatası: {e}")
        return None, None

    # --- BİRLEŞTİRME VE MAPPING OLUŞTURMA ---
    full_sequence = ""
    mapping = {}
    global_index = 0 # ESM-2 vektöründeki satır numarası (0, 1, 2...)

    # Zincirleri, dosyada gördüğümüz sırayla birleştiriyoruz (Kritik!)
    for chain in chain_order:
        # 1. O zincirin harflerini ana diziye ekle
        seq_segment = "".join(sequences[chain])
        full_sequence += seq_segment
        
        # 2. O zincirdeki residue'ları sırasıyla global indekse eşle
        for res_key in chain_res_info[chain]:
            # res_key: ('A', 45) gibi bir tuple
            mapping[res_key] = global_index
            global_index += 1
            
    if not full_sequence:
        return None, None
        
    return full_sequence, mapping

# --- compute_esm2_embeddings (DİKKAT: Büyük harf hatası potansiyeli) ---
# ESM-2 hesaplama fonksiyonunda şu satırı kontrol edin:
# ESM-2 HDF5 anahtarları BÜYÜK harfli olduğu için PDB ID'yi BÜYÜK harf yapmalısınız.
# Bu düzeltme zaten var: if pdb_id.upper() not in processed_pdb_ids: 

# Bu düzeltmelerle, "Toplam 0 adet protein dizisi çıkarıldı" hatasını aşmanız gerekir.

def extract_all_sequences(map_file: str) -> List[Tuple[str, str]]:
    """Adım 1: JSON haritasından tüm PDB ID'leri ve dizileri çıkarır."""
    print("\n--- ADIM 1: FASTA Dizileri Çıkarılıyor ---")
    
    try:
        with open(map_file, 'r') as f:
            data_map = json.load(f)
    except FileNotFoundError:
        print(f"HATA: Harita dosyası bulunamadı: {map_file}")
        return []

    sequences = []
    
    # İlerleme çubuğunu kullanmak için listeyi hazırlayın
    pdb_items = list(data_map.items())
    for pdb_id, data in tqdm(pdb_items, desc="FASTA dizileri işleniyor"):
        prepared_pdb_path = data.get('protein_pdb')
        
        if not prepared_pdb_path:
            continue
            
        # --- DÜZELTME BURADA ---
        # Fonksiyon iki değer döndürüyor, biz sadece ilkiyle (sequence) ilgileniyoruz.
        # İkincisini (mapping) _ ile yoksayıyoruz.
        sequence_tuple = parse_fasta_from_pdb(prepared_pdb_path)
        
        if sequence_tuple is None:
            continue
            
        sequence, _ = sequence_tuple # Tuple Unpacking (Ayıklama)
        
        # Artık sequence bir String olduğu için len() doğru çalışır
        if sequence and len(sequence) > 10:
            sequences.append((pdb_id, sequence))
            
    print(f"✅ Adım 1 Başarılı: Toplam {len(sequences)} adet protein dizisi çıkarıldı.")
    return sequences

def compute_esm2_embeddings(sequences: List[Tuple[str, str]], h5_file: str):
    """Adım 2: Çıkarılan diziler için ESM-2 gömülerini hesaplar ve HDF5'e kaydeder."""
    
    if not sequences:
        print("HATA: Hesaplama için dizi bulunamadı.")
        return

    print(f"\n--- ADIM 2: ESM-2 Gömüleri Hesaplama ({MODEL_NAME}) ---")
    
    # 1. Modeli ve Alfabe'yi Yükle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("UYARI: GPU (CUDA) bulunamadı. Hesaplama CPU'da yapılacak ve ÇOK yavaş olacaktır.")

    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
        model = model.to(device)
        model.eval()
        print(f"Model başarıyla {device}'a yüklendi. Gömü Boyutu: {EMBEDDING_DIM}")
    except Exception as e:
        print(f"KRİTİK HATA: ESM-2 modeli yüklenemedi: {e}")
        return

    batch_converter = alphabet.get_batch_converter()

    # 2. HDF5 Dosyasını Hazırlama ve Kaldığı Yerden Devam Etme Mantığı
    
    # Başarıyla işlenmiş PDB ID'lerini tutacak set
    processed_pdb_ids = set()
    h5_file_exists = os.path.exists(h5_file)
    
    if h5_file_exists:
        try:
            # H5 dosyasını oku kipinde açıp içindeki anahtarları (PDB ID'lerini) al
            with h5py.File(h5_file, 'r') as h5f_read:
                processed_pdb_ids.update(h5f_read.keys())
            
            print(f"UYARI: Mevcut H5 dosyası bulundu. {len(processed_pdb_ids)} kayıt atlanacak.")
            
        except Exception as e:
            tqdm.write(f"KRİTİK HATA: Mevcut H5 dosyası ({h5_file}) bozuk görünüyor ({e}). Baştan başlanıyor.")
            h5_file_exists = False # Dosya bozuksa baştan başlat
            
    # Filtreleme: Yalnızca henüz işlenmemiş dizileri al
    sequences_to_process = []
    for pdb_id, seq in sequences:
        if pdb_id.upper() not in processed_pdb_ids: # HDF5 anahtarları BÜYÜK harfli
            sequences_to_process.append((pdb_id, seq))

    if not sequences_to_process:
        print("✅ Tüm kayıtlar zaten işlenmiş. İşlem tamamlandı.")
        return

    print(f"İşlenmemiş toplam {len(sequences_to_process)} dizi bulundu.")
    
    # H5 dosyasını açma kipi: Varsa ekleme ('a'), yoksa yazma ('w')
    h5_open_mode = 'a' if h5_file_exists else 'w'
    
    # 3. HDF5 Dosyasını Açma ve İşleme Başlatma
    # PDB ID listesini sadece yeni başladığında ('w') veya ekleme ('a') kipinde de açabiliriz.
    with h5py.File(h5_file, h5_open_mode) as h5f, open(PDB_ID_LIST_FILE, 'a' if h5_file_exists else 'w') as pdb_list_f:
        print(f"'{h5_file}' dosyasına {h5_open_mode} kipiyle yazma işlemi başlıyor.")
        
        start_total_time = time.time()
        
        # tqdm döngüsünü sequences_to_process üzerinde çalıştır
        for i in tqdm(range(0, len(sequences_to_process), BATCH_SIZE), desc="ESM-2 Gömüleri Hesaplama"):
            
            batch_data = sequences_to_process[i:i+BATCH_SIZE] # <--- DİKKAT: Artık sequences_to_process kullanılıyor
            
            # PDB ID'ler, diziler ve token'lar
            batch_pdb_ids = [item[0].upper() for item in batch_data] 
            batch_seqs = [item[1] for item in batch_data]

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            # 3. Gömüleri Hesapla
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[REPR_LAYER], return_contacts=False)
            
            representations = results["representations"][REPR_LAYER].cpu() 
            
            # 4. HDF5'e Kaydet
            for j in range(representations.shape[0]):
                pdb_id = batch_pdb_ids[j]
                seq_len = len(batch_seqs[j])
                
                # [CLS] (Başlangıç) ve [EOS] (Bitiş) token'larını atla
                embedding = representations[j, 1 : seq_len + 1] # Boyut: [seq_len, 1280]
                
                if embedding.shape[0] != seq_len or embedding.shape[1] != EMBEDDING_DIM:
                    tqdm.write(f"UYARI: {pdb_id} için boyut uyuşmazlığı ({embedding.shape}). Atlanıyor.")
                    continue

                try:
                    h5f.create_dataset(pdb_id, data=embedding.numpy(), compression="gzip")
                    pdb_list_f.write(pdb_id + '\n')
                except RuntimeError as e:
                    # Bu genellikle "Name already exists" hatasıdır. Atlama mantığı çalışmış demektir.
                    tqdm.write(f"UYARI: {pdb_id} zaten kaydedilmiş görünüyor. Atlama.")
                except Exception as write_error:
                    tqdm.write(f"KRİTİK YAZMA HATASI ({pdb_id}): {write_error}")
                    # Hata durumunda işlemi sonlandırıp dosya sistemini kontrol edin.
                    # Eğer I/O hatası (errno=5) tekrar gelirse, bu dosya sistemidir.
                    return
        
        # ... (Süre yazdırma ve başarılı mesajları) ...
        print(f"\n✅ Adım 2 Başarılı: İşlem tamamlandı. Toplam süre: {(time.time() - start_total_time) / 60:.2f} dakika.")
        

def main():
    parser = argparse.ArgumentParser(description="PDBbind için FASTA çıkarımı ve ESM-2 Gömü Hesaplamasını tek bir adımda gerçekleştirir.")
    parser.add_argument('-m', '--map_file', default=DEFAULT_INPUT_JSON, help="Girdi JSON harita dosyası.")
    args = parser.parse_args()

    # Adım 1'i Çalıştır
    sequences = extract_all_sequences(args.map_file)

    # Adım 2'yi Çalıştır
    if sequences:
        compute_esm2_embeddings(sequences, OUTPUT_H5_FILE)
    else:
        print("İşlem dizisi boş olduğu için ESM-2 hesaplaması atlandı.")


if __name__ == "__main__":
    main()