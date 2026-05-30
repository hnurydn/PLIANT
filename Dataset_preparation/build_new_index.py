"""
build_new_index.py
==================
train_meta.json / valid_meta.json / core_meta.json dosyalarından
yeni format index pkl'leri oluşturur.

Yeni index yapısı (pdb_id bazlı):
{
  "2ydo": {"pK": 7.2,  "binder_key": "binder_000012345", "decoy_keys": ["decoy_000012346", ...]},
  "1abc": {"pK": 8.1,  "binder_key": "binder_000054321", "decoy_keys": ["decoy_000054322"]},
  ...
}

Bu sayede ProLigDataset:
  - Batch oluştururken sadece key listelerini kullanır
  - LMDB'den sadece seçilen key'leri çeker (tek decoy ~10KB, 237MB değil)
  - Her epoch'ta farklı random decoy seçimi yapılır

Kullanım:
    python build_new_index.py \
        --meta_dir  /arf/scratch/.../lmdb_planet \
        --out_dir   /arf/scratch/.../index
"""

import json, pickle, argparse, os
from collections import defaultdict
from tqdm import tqdm


def build_index_from_meta(meta_path: str, out_path: str):
    set_name = os.path.basename(meta_path).replace("_meta.json", "")
    print(f"\n{'='*60}")
    print(f"  {set_name.upper()} işleniyor: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # pdb_id bazında grupla
    index = defaultdict(lambda: {"pK": 0.0, "binder_key": None, "decoy_keys": []})

    for record in tqdm(meta, desc=f"  Gruplama ({set_name})"):
        pdb_id = record["pdb_id"]
        key    = record["key"]

        if record["is_decoy"]:
            index[pdb_id]["decoy_keys"].append(key)
        else:
            index[pdb_id]["binder_key"] = key
            index[pdb_id]["pK"]         = record["pK"]

    # Binder'ı olmayan pdb'leri çıkar (incomplete kayıtlar)
    clean_index = {
        pdb_id: info
        for pdb_id, info in index.items()
        if info["binder_key"] is not None
    }

    # Özet
    total_binders = len(clean_index)
    total_decoys  = sum(len(v["decoy_keys"]) for v in clean_index.values())
    no_decoy      = sum(1 for v in clean_index.values() if not v["decoy_keys"])

    print(f"  pdb sayısı     : {total_binders}")
    print(f"  toplam binder  : {total_binders}")
    print(f"  toplam decoy   : {total_decoys}")
    print(f"  decoy'suz pdb  : {no_decoy}")
    print(f"  ort. decoy/pdb : {total_decoys / max(total_binders,1):.1f}")

    with open(out_path, "wb") as f:
        pickle.dump(clean_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  ✅ Kaydedildi  : {out_path}")
    return clean_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", required=True,
                        help="train_meta.json, valid_meta.json, core_meta.json dizini")
    parser.add_argument("--out_dir",  required=True,
                        help="Çıktı pkl'lerinin yazılacağı dizin (index klasörü)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mapping = {
        "train_meta.json": "train_esm2.pkl",
        "valid_meta.json": "valid_esm2.pkl",
        "core_meta.json":  "core_esm2.pkl",
    }

    for meta_file, out_file in mapping.items():
        meta_path = os.path.join(args.meta_dir, meta_file)
        out_path  = os.path.join(args.out_dir,  out_file)

        if not os.path.exists(meta_path):
            print(f"⚠️  Bulunamadı, atlandı: {meta_path}")
            continue

        build_index_from_meta(meta_path, out_path)

    print("\n🎉 Tüm index'ler hazır!")


if __name__ == "__main__":
    main()
