"""
pretensorize_lmdb.py
====================
Mevcut LMDB'deki her ComplexPocket objesini tensorize edip
yeni bir LMDB'ye yazar.

Kullanım:
    python pretensorize_lmdb.py --input train.lmdb --output train_tensor.lmdb --njobs 40
"""

import sys, os, pickle, lmdb, json, argparse
import torch
from multiprocessing import Pool
from tqdm import tqdm

sys.path.insert(0, '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation')
from chemutils_esm2_vector import tensorize_all

MAP_SIZE_GB = 100


def process_one(args):
    """Tek bir (key, raw_bytes, is_decoy) üçlüsünü tensorize eder."""
    key, raw_bytes, is_decoy = args
    try:
        obj = pickle.loads(raw_bytes)
        obj.pK = 0.0 if is_decoy else obj.pK

        outputs = tensorize_all([obj], decoy_flag=False)

        # Tüm tensor/tuple/list'i pickle ile sakla
        serialized = pickle.dumps(outputs, protocol=pickle.HIGHEST_PROTOCOL)
        return key, serialized, None
    except Exception as e:
        return key, None, str(e)


def pretensorize(input_lmdb, output_lmdb, meta_path, njobs):
    # 1. Meta JSON oku
    input_meta_path = input_lmdb.replace('.lmdb', '_meta.json').replace(
        os.path.basename(input_lmdb),
        os.path.basename(input_lmdb).replace('.lmdb', '_meta.json')
    )
    # Meta path: aynı klasörde set_name_meta.json
    base = os.path.splitext(input_lmdb)[0]
    input_meta_path = base + '_meta.json'

    with open(input_meta_path, 'r') as f:
        meta = json.load(f)

    print(f"Toplam kayıt: {len(meta)}")

    # 2. Input LMDB aç
    env_in = lmdb.open(input_lmdb, readonly=True, lock=False,
                       readahead=True, meminit=False)

    # 3. Tüm kayıtları oku
    print("LMDB'den veriler okunuyor...")
    tasks = []
    with env_in.begin() as txn:
        for record in tqdm(meta, desc="Okuma"):
            key = record['key'].encode()
            raw = txn.get(key)
            if raw is None:
                continue
            tasks.append((record['key'], raw, record['is_decoy']))
    env_in.close()

    print(f"Okundu: {len(tasks)} kayıt. Tensorize başlıyor...")

    # 4. Output LMDB aç
    env_out = lmdb.open(output_lmdb, map_size=MAP_SIZE_GB * 1024**3,
                        subdir=True, meminit=False, map_async=True, writemap=True)

    # 5. Paralel tensorize
    errors = 0
    new_meta = []

    with Pool(processes=njobs) as pool:
        with env_out.begin(write=True) as txn:
            for key, serialized, err in tqdm(
                pool.imap_unordered(process_one, tasks, chunksize=4),
                total=len(tasks), desc="Tensorize"
            ):
                if err:
                    errors += 1
                    if errors <= 10:
                        print(f"\n⚠️  Hata [{key}]: {err}")
                    continue

                txn.put(key.encode(), serialized)

                # Meta kaydını koru
                orig = next((r for r in meta if r['key'] == key), None)
                if orig:
                    new_meta.append(orig)

    env_out.close()

    # 6. Yeni meta kaydet
    out_meta_path = os.path.splitext(output_lmdb)[0] + '_meta.json'
    with open(out_meta_path, 'w') as f:
        json.dump(new_meta, f)

    binders = sum(1 for r in new_meta if not r['is_decoy'])
    decoys  = sum(1 for r in new_meta if r['is_decoy'])
    print(f"\n✅ Tamamlandı!")
    print(f"   Binder: {binders} | Decoy: {decoys} | Hata: {errors}")
    print(f"   Çıktı LMDB:  {output_lmdb}")
    print(f"   Çıktı Meta:  {out_meta_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help="Girdi LMDB klasörü")
    parser.add_argument('--output', required=True, help="Çıktı LMDB klasörü")
    parser.add_argument('--njobs',  type=int, default=40)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pretensorize(args.input, args.output, None, args.njobs)
