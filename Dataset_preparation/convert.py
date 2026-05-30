"""
build_lmdb_from_pickles.py
"""

import os, pickle, lmdb, json, argparse
from tqdm import tqdm

INDEX_DIR   = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index'
MAP_SIZE_GB = 500

def build_set(set_name, out_dir):
    index_pkl = os.path.join(INDEX_DIR, f'{set_name}.pkl')
    out_lmdb  = os.path.join(out_dir, f'{set_name}.lmdb')
    out_meta  = os.path.join(out_dir, f'{set_name}_meta.json')

    print(f'\n{"="*60}')
    print(f'  {set_name.upper()} işleniyor...')

    with open(index_pkl, 'rb') as f:
        records = pickle.load(f)

    binder_records = [(path, pk) for path, pk, is_decoy in records if not is_decoy]
    decoy_records  = [(path, pk) for path, pk, is_decoy in records if is_decoy]

    print(f'  Binder kayıt : {len(binder_records)}')
    print(f'  Decoy kayıt  : {len(decoy_records)}')

    # Sadece path'leri sakla, şimdi yükleme yok
    decoy_path_map = {}
    for path, _ in decoy_records:
        pdb_id = os.path.basename(path).replace('_decoys.pkl', '')
        decoy_path_map[pdb_id] = path

    print(f'  Decoy path map hazır: {len(decoy_path_map)} PDB')

    # LMDB oluştur
    os.makedirs(out_lmdb, exist_ok=True)
    env = lmdb.open(out_lmdb, map_size=MAP_SIZE_GB * 1024**3,
                    subdir=True, meminit=False, map_async=True)

    new_meta = []
    errors   = 0

    with env.begin(write=True) as txn:
        for path, pk in tqdm(binder_records, desc=f'  Binder yazma'):
            pdb_id = os.path.basename(path).replace('_binder.pkl', '')
            key    = pdb_id

            try:
                # Binder yükle
                with open(path, 'rb') as f:
                    binder_obj = pickle.load(f)

                # Decoy pkl'yi o anda yükle
                decoy_path = decoy_path_map.get(pdb_id)
                mols = []
                if decoy_path:
                    try:
                        with open(decoy_path, 'rb') as f:
                            decoy_list = pickle.load(f)
                        if isinstance(decoy_list, list):
                            for obj in decoy_list:
                                if hasattr(obj, 'ligand') and obj.ligand is not None:
                                    mols.append(obj.ligand)
                                elif hasattr(obj, 'atoms'):
                                    mols.append(obj)
                    except Exception as e:
                        print(f'\n  ⚠️  Decoy yükleme hatası [{pdb_id}]: {e}')

                binder_obj.decoys       = mols
                binder_obj.decoys_count = len(mols)
                binder_obj.pK           = pk

                txn.put(
                    key.encode(),
                    pickle.dumps(binder_obj, protocol=pickle.HIGHEST_PROTOCOL)
                )

                new_meta.append({
                    'key'     : key,
                    'pK'      : pk,
                    'is_decoy': False,
                    'pdb_id'  : pdb_id,
                    'n_decoys': len(mols),
                })

            except Exception as e:
                errors += 1
                print(f'\n  ⚠️  Hata [{pdb_id}]: {e}')
                continue

    env.sync()
    env.close()

    with open(out_meta, 'w') as f:
        json.dump(new_meta, f)

    avg_decoys = sum(m['n_decoys'] for m in new_meta) / max(len(new_meta), 1)
    no_decoy   = sum(1 for m in new_meta if m['n_decoys'] == 0)
    print(f'  ✅ Kaydedildi     : {out_lmdb}')
    print(f'  Toplam binder     : {len(new_meta)}')
    print(f'  Ort. decoy/binder : {avg_decoys:.1f}')
    print(f'  Decoy\'suz binder  : {no_decoy}')
    print(f'  Hata              : {errors}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_lmdb', required=True)
    parser.add_argument('--sets', nargs='+', default=['train', 'valid', 'core'])
    args = parser.parse_args()

    os.makedirs(args.out_lmdb, exist_ok=True)

    for set_name in args.sets:
        build_set(set_name, args.out_lmdb)

    print('\n🎉 Tüm setler tamamlandı!')

if __name__ == '__main__':
    main()