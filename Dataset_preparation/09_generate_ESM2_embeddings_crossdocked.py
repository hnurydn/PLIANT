"""
09_generate_ESM2_embeddings_crossdocked.py
==========================================
Generates ESM-2 embeddings from CrossDocked 2020 receptor PDB files.
APPENDS to the existing H5 file (does not overwrite it).

- PDB IDs already present in the H5 file are skipped
- CrossDocked PDB IDs are lowercase, H5 keys are uppercase -> converted to
  uppercase

Usage:
    python 09_generate_ESM2_embeddings_crossdocked.py \
        --json_path $PLANET_ROOT/.../datasets/crossdocked_new_records.json \
        --extracted_dir $PLANET_ROOT/.../datasets/crossdocked_extracted \
        --h5_path $PLANET_ROOT/.../pdbbind2020_esm2_embeddings_v2.h5 \
        --batch_size 2
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import esm
import h5py
from tqdm import tqdm

# ── Sabitler ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "esm2_t33_650M_UR50D"
REPR_LAYER    = 33
EMBEDDING_DIM = 1280

AA_3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V'
}

# ── PDB → sequence ────────────────────────────────────────────────────────────
def parse_sequence_from_pdb(pdb_path):
    """PDB dosyasından tek zincir (birleşik) amino asit dizisi çıkarır."""
    sequences   = {}
    chain_order = []
    current_res = None

    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                chain_id    = line[21].strip() or 'A'
                res_num_str = line[22:26].strip()
                res_type    = line[17:20].strip()

                if not res_num_str or res_type not in AA_3_TO_1:
                    continue

                res_key = (chain_id, int(res_num_str))
                if chain_id not in sequences:
                    sequences[chain_id] = []
                    chain_order.append(chain_id)

                if res_key != current_res:
                    one_letter = AA_3_TO_1[res_type]
                    sequences[chain_id].append(one_letter)
                    current_res = res_key

    except Exception as e:
        return None

    full_seq = "".join("".join(sequences[c]) for c in chain_order)
    return full_seq if len(full_seq) > 10 else None


# ── Receptor PDB yollarını bul ────────────────────────────────────────────────
def build_pdb_map(json_path, extracted_dir):
    """
    JSON'daki her kayıt için pdb_id → receptor PDB dosyası eşleşmesini kurar.
    Aynı pdb_id birden fazla kayıtta olabilir — tekil tutuyoruz.
    """
    import glob
    d = json.load(open(json_path))

    pdb_to_path = {}
    missing     = 0

    for r in d:
        pdb_id = r['pdb_id']
        if pdb_id in pdb_to_path:
            continue  # zaten bulunmuş

        folder   = r['receptor'].split('/')[0]   # örn: 1433B_HUMAN_1_240_pep_0
        folder_path = os.path.join(extracted_dir, folder)

        # Önce <pdb_id>_A_rec.pdb dene, sonra glob ile herhangi bir rec.pdb
        pdb_file = os.path.join(folder_path, f'{pdb_id}_A_rec.pdb')
        if not os.path.exists(pdb_file):
            matches = glob.glob(os.path.join(folder_path, f'{pdb_id}_*.pdb'))
            pdb_file = matches[0] if matches else None

        if pdb_file and os.path.exists(pdb_file):
            pdb_to_path[pdb_id] = pdb_file
        else:
            missing += 1

    print(f"✅ Receptor eşleştirme: {len(pdb_to_path)} bulundu, {missing} bulunamadı")
    return pdb_to_path


# ── Embedding üretimi ─────────────────────────────────────────────────────────
def generate_embeddings(pdb_to_path, h5_path, batch_size=2):
    """H5'te olmayan proteinler için ESM2 embedding üretir ve ekler."""

    # Mevcut H5 keylerini oku
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as h5f:
            existing_keys = set(h5f.keys())
        print(f"H5'te mevcut: {len(existing_keys)} kayıt")
    else:
        existing_keys = set()
        print("H5 dosyası yok, yeni oluşturulacak")

    # H5'te olmayan PDB ID'leri filtrele (uppercase karşılaştırma)
    to_process = [
        (pdb_id, path)
        for pdb_id, path in pdb_to_path.items()
        if pdb_id.upper() not in existing_keys
    ]
    print(f"İşlenecek yeni protein: {len(to_process)}")

    if not to_process:
        print("✅ Tüm embedding'ler zaten mevcut, işlem yapılmadı.")
        return

    # ESM2 modelini yükle
    print(f"\nESM2 modeli yükleniyor: {MODEL_NAME}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("✅ Model yüklendi")

    # Sequence'ları çıkar
    print("\nSequence'lar çıkarılıyor...")
    entries = []
    for pdb_id, pdb_path in tqdm(to_process, desc="PDB okuma"):
        seq = parse_sequence_from_pdb(pdb_path)
        if seq:
            entries.append((pdb_id.upper(), seq))  # H5'e BÜYÜK harfle yaz
        else:
            tqdm.write(f"  ⚠️  Sequence çıkarılamadı: {pdb_id}")

    print(f"✅ {len(entries)} geçerli sequence")

    # Batch halinde embedding üret
    saved = 0
    t0 = time.time()

    with h5py.File(h5_path, 'a') as h5f:
        for i in tqdm(range(0, len(entries), batch_size), desc="ESM2 embedding"):
            batch = entries[i : i + batch_size]
            batch_input = [(h5_key, seq) for h5_key, seq in batch]

            try:
                _, _, tokens = batch_converter(batch_input)
                tokens = tokens.to(device)

                with torch.no_grad():
                    results = model(tokens, repr_layers=[REPR_LAYER],
                                    return_contacts=False)

                representations = results['representations'][REPR_LAYER].cpu()

                for j, (h5_key, seq) in enumerate(batch):
                    seq_len   = len(seq)
                    embedding = representations[j, 1:seq_len + 1].numpy()

                    if embedding.shape != (seq_len, EMBEDDING_DIM):
                        tqdm.write(f"  ⚠️  Boyut hatası [{h5_key}]: {embedding.shape}")
                        continue

                    if h5_key in h5f:
                        tqdm.write(f"  ⚠️  Zaten var (atlandı): {h5_key}")
                        continue

                    h5f.create_dataset(h5_key, data=embedding, compression='gzip')
                    saved += 1

            except Exception as e:
                tqdm.write(f"  ❌ Batch hatası: {e}")
                continue

    elapsed = (time.time() - t0) / 60
    print(f"\n✅ {saved} yeni embedding kaydedildi ({elapsed:.1f} dk)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='CrossDocked receptor PDB dosyaları için ESM2 embedding üretir')
    parser.add_argument('--json_path',      required=True,
                        help='crossdocked_new_records.json yolu')
    parser.add_argument('--extracted_dir',  required=True,
                        help='crossdocked_extracted klasörü')
    parser.add_argument('--h5_path',        required=True,
                        help='Mevcut H5 dosyası (eklenecek)')
    parser.add_argument('--batch_size',     type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("CrossDocked ESM2 Embedding Üretimi")
    print("=" * 60)

    # 1. Receptor PDB eşleştirmesi
    pdb_to_path = build_pdb_map(args.json_path, args.extracted_dir)

    # 2. Embedding üret ve H5'e ekle
    generate_embeddings(pdb_to_path, args.h5_path, args.batch_size)

    print("\n✅ Tüm işlem tamamlandı.")


if __name__ == '__main__':
    main()