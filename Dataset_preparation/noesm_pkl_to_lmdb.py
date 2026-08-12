"""
noesm_pkl_to_lmdb.py
====================
Writes pkl files for the noESM models into an LMDB.
ComplexPocket objects are stored as-is (not tensorized).
Tensorization happens in a later, separate step.

Usage:
    python noesm_pkl_to_lmdb.py
"""

import sys, pickle, lmdb, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# ─── AYARLAR ──────────────────────────────────────────────────────────────────
INDEX_DIR   = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/index")
OUT_DIR     = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/lmdb_noesm")
MAP_SIZE_GB = 350
MAX_DECOY   = 100
# ──────────────────────────────────────────────────────────────────────────────

SETS = {
    'train': 'train.pkl',
    'valid': 'valid.pkl',
    'core':  'core.pkl',
}

os.makedirs(OUT_DIR, exist_ok=True)


def extract_pdb_id(source_path):
    fname  = os.path.basename(source_path)  # 4f3c_binder.pkl
    pdb_id = fname.split('_')[0]            # 4f3c
    return pdb_id


for set_name, pkl_file in SETS.items():
    pkl_path = os.path.join(INDEX_DIR, pkl_file)
    if not os.path.exists(pkl_path):
        print(f'\n⚠️  {pkl_path} yok, atlanıyor.')
        continue

    print(f'\n{"="*50}')
    print(f'  {set_name.upper()}')
    print(f'{"="*50}')

    with open(pkl_path, 'rb') as f:
        index = pickle.load(f)

    out_lmdb  = os.path.join(OUT_DIR, f'{set_name}_noesm.lmdb')
    meta      = []
    errors    = 0
    entry_idx = 0

    env = lmdb.open(out_lmdb, map_size=MAP_SIZE_GB * 1024**3,
                    subdir=True, meminit=False, map_async=True, writemap=True)

    with env.begin(write=True) as txn:
        for record in tqdm(index, desc='  yazılıyor'):
            source_path = record[0]
            pk_val      = float(record[1])
            is_multi    = bool(record[2]) if len(record) > 2 else False
            pdb_id      = extract_pdb_id(source_path)

            try:
                with open(source_path, 'rb') as f:
                    content = pickle.load(f)

                if is_multi:
                    # Decoy: liste içinde direkt ComplexPocket objeleri
                    entries = content if isinstance(content, list) else [content]
                    entries = entries[:MAX_DECOY]

                    for obj in entries:
                        obj.pK = 0.0
                        key = f'decoy_{entry_idx:09d}'.encode()
                        txn.put(key, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
                        meta.append({
                            'key':      key.decode(),
                            'pK':       0.0,
                            'is_decoy': True,
                            'pdb_id':   pdb_id,
                        })
                        entry_idx += 1
                else:
                    # Binder: tek ComplexPocket objesi
                    content.pK = pk_val
                    key = f'binder_{entry_idx:09d}'.encode()
                    txn.put(key, pickle.dumps(content, protocol=pickle.HIGHEST_PROTOCOL))
                    meta.append({
                        'key':      key.decode(),
                        'pK':       pk_val,
                        'is_decoy': False,
                        'pdb_id':   pdb_id,
                    })
                    entry_idx += 1

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f'\n  ⚠️  Atlandı: {os.path.basename(source_path)} → {e}')

    env.close()

    meta_path = os.path.join(OUT_DIR, f'{set_name}_noesm_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    binders = sum(1 for r in meta if not r['is_decoy'])
    decoys  = sum(1 for r in meta if r['is_decoy'])
    print(f'  ✅ Binder: {binders}  |  Decoy: {decoys}  |  Hata: {errors}')
    print(f'  📁 {out_lmdb}')
    print(f'  📄 {meta_path}')

print('\n🎉 Tamamlandı!')