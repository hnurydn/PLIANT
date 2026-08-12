import os
import argparse
import time
import numpy as np
import torch
import esm
import h5py
from tqdm import tqdm
from Bio import SeqIO # BioPython gerekli
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- AYARLAR ---
DEFAULT_FASTA = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_sequences_for_esm2.fasta")
OUTPUT_H5_FILE = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/dude/dude_esm2_embeddings.h5")

# ESM-2 650M Model
MODEL_NAME = "esm2_t33_650M_UR50D"
REPR_LAYER = 33 
EMBEDDING_DIM = 1280 
BATCH_SIZE = 1 # Bellek güvenliği için 1 (Daha hızlı olsun dersen 2 yapabilirsin)

def get_clean_id(header):
    """FASTA başlığından temiz target ismini çıkarır."""
    # Örn: "DUDE_aa2ar" -> "aa2ar"
    # Örn: "LITPCBA_ADRB2" -> "ADRB2"
    if header.startswith("DUDE_"):
        return header.replace("DUDE_", "")
    elif header.startswith("LITPCBA_"):
        return header.replace("LITPCBA_", "")
    return header

def run_esm2_generation(fasta_path, h5_path):
    print(f"\n🚀 ESM-2 Embedding Üretimi Başlıyor")
    print(f"📄 Girdi FASTA: {fasta_path}")
    print(f"💾 Çıktı H5: {h5_path}")

    # 1. Modeli Yükle
    print(f"🔄 Model Yükleniyor: {MODEL_NAME}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
        model = model.to(device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        print(f"✅ Model {device} üzerinde hazır.")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return

    # 2. FASTA'yı Oku
    sequences = []
    print("📖 FASTA dosyası okunuyor...")
    if not os.path.exists(fasta_path):
        print("❌ FASTA dosyası bulunamadı!")
        return

    for record in SeqIO.parse(fasta_path, "fasta"):
        clean_key = get_clean_id(record.id)
        seq_str = str(record.seq)
        # Çok kısa sekansları ele (ESM bazen 0 uzunlukta hata verir)
        if len(seq_str) > 5: 
            sequences.append((clean_key, seq_str))
    
    print(f"📊 Toplam {len(sequences)} sekans bulundu.")

    # 3. H5 Dosyasını Hazırla (Resume Logic)
    processed_ids = set()
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            processed_ids.update(f.keys())
        print(f"⚠️ {len(processed_ids)} kayıt zaten mevcut, bunlar atlanacak.")

    # İşlenecekleri filtrele
    to_process = [x for x in sequences if x[0] not in processed_ids]
    
    if not to_process:
        print("✅ Tüm sekanslar zaten işlenmiş!")
        return

    # 4. Döngü
    print(f"⚙️ {len(to_process)} yeni sekans işlenecek...")
    
    # H5 dosyasını 'a' (append) modunda açıyoruz
    with h5py.File(h5_path, 'a') as h5f:
        
        for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Embedding"):
            batch = to_process[i : i + BATCH_SIZE]
            
            # Format: [(id, seq), (id, seq)]
            batch_labels = [item[0] for item in batch]
            batch_seqs = [item[1] for item in batch]
            
            # Tokenize
            # data format for converter: [(label, seq), ...]
            data = batch 
            labels, strs, tokens = batch_converter(data)
            tokens = tokens.to(device)

            # Inference
            with torch.no_grad():
                # Uzun sekanslarda OOM (Out of Memory) olmamak için
                # Eğer sekans çok uzunsa (örn > 2000), bunu try-except ile yakalayabiliriz
                try:
                    results = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n⚠️ OOM Hatası (Batch atlanıyor): {batch_labels}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            token_representations = results["representations"][REPR_LAYER]

            # Kaydetme
            for j, pdb_id in enumerate(batch_labels):
                # Başlangıç ve bitiş tokenlarını kes (1 : len+1)
                seq_len = len(batch_seqs[j])
                embedding = token_representations[j, 1 : seq_len + 1].cpu().numpy().astype(np.float16)
                
                try:
                    h5f.create_dataset(pdb_id, data=embedding, compression="gzip")
                except ValueError:
                    # Nadir durumlarda aynı anda yazma çakışması olursa
                    print(f"Skipping duplicate: {pdb_id}")

            # Bellek Temizliği (Opsiyonel ama güvenli)
            del tokens, results, token_representations
            torch.cuda.empty_cache()

    print(f"\n✅ Tüm işlemler bitti. Dosya: {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', default=DEFAULT_FASTA, help="Girdi FASTA dosyası")
    parser.add_argument('-o', '--output', default=OUTPUT_H5_FILE, help="Çıktı H5 dosyası")
    args = parser.parse_args()

    run_esm2_generation(args.fasta, args.output)