"""
update_lmdb_decoys.py
Belirtilen PDB listesi için LMDB'deki binder objelerinin decoy listesini günceller.
"""
import os, pickle, lmdb, json, argparse
from tqdm import tqdm

ALL_PICKLES_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/all_pickles_2'

def update_set(set_name, lmdb_dir, target_pdb_ids):
    lmdb_path = os.path.join(lmdb_dir, f'{set_name}.lmdb')
    meta_path = os.path.join(lmdb_dir, f'{set_name}_meta.json')

    if not os.path.exists(lmdb_path):
        print(f'⚠️  {set_name}.lmdb bulunamadı, atlanıyor.')
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Bu set'teki hedef PDB'leri bul
    set_pdb_ids = {m['pdb_id'] for m in meta}
    to_update = set_pdb_ids & target_pdb_ids
    print(f'\n{set_name.upper()}: {len(to_update)} PDB güncellenecek')

    if not to_update:
        return

    env = lmdb.open(lmdb_path, map_size=500 * 1024**3,
                    readonly=False, lock=False, meminit=False)

    updated = 0
    skipped = 0

    with env.begin(write=True) as txn:
        for pdb_id in tqdm(to_update, desc=f'  {set_name} güncelleniyor'):
            decoy_pkl = os.path.join(ALL_PICKLES_DIR, f'{pdb_id}_decoys.pkl')
            if not os.path.exists(decoy_pkl):
                skipped += 1
                continue

            # Mevcut binder objesini oku
            raw = txn.get(pdb_id.encode())
            if raw is None:
                skipped += 1
                continue

            try:
                binder_obj = pickle.loads(bytes(raw))

                # Decoy pkl'yi yükle
                with open(decoy_pkl, 'rb') as f:
                    decoy_list = pickle.load(f)

                mols = []
                for obj in decoy_list:
                    if hasattr(obj, 'ligand') and obj.ligand is not None:
                        mols.append(obj.ligand)
                    elif hasattr(obj, 'atoms'):
                        mols.append(obj)

                binder_obj.decoys = mols
                binder_obj.decoys_count = len(mols)

                txn.put(pdb_id.encode(),
                        pickle.dumps(binder_obj, protocol=pickle.HIGHEST_PROTOCOL))
                updated += 1

            except Exception as e:
                print(f'  ⚠️  {pdb_id} hatası: {e}')
                skipped += 1

    env.sync()
    env.close()

    # Meta JSON'u güncelle
    for m in meta:
        if m['pdb_id'] in to_update:
            decoy_pkl = os.path.join(ALL_PICKLES_DIR, f"{m['pdb_id']}_decoys.pkl")
            if os.path.exists(decoy_pkl):
                try:
                    with open(decoy_pkl, 'rb') as f:
                        decoy_list = pickle.load(f)
                    m['n_decoys'] = len([o for o in decoy_list
                                         if hasattr(o, 'ligand') and o.ligand is not None])
                except:
                    pass

    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print(f'  ✅ Güncellendi: {updated}, Atlandı: {skipped}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_dir', required=True, help='LMDB dizini')
    parser.add_argument('--sdf_dir',  required=True, help='Yeni SDF dizini')
    parser.add_argument('--sets', nargs='+', default=['train', 'valid', 'core'])
    args = parser.parse_args()

    # Hedef PDB listesini SDF dosyalarından al
    target_pdb_ids = set(
        f.replace('_decoys.sdf', '')
        for f in os.listdir(args.sdf_dir)
        if f.endswith('_decoys.sdf')
    )
    print(f'Hedef PDB sayısı: {len(target_pdb_ids)}')

    for set_name in args.sets:
        update_set(set_name, args.lmdb_dir, target_pdb_ids)

    print('\n🎉 Tamamlandı!')

if __name__ == '__main__':
    main()