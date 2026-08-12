"""
update_lmdb_with_esm2.py
========================
lmdb_planet içindeki binder objelerine ESM2 embedding ekler.
Her PDB için sadece 1 embedding — decoy'lar aynı proteini paylaşır.

Giriş : lmdb_planet  (binder_key → ComplexPocket)
Çıkış : lmdb_planet_v2  (binder_key → ComplexPocket + res_features)
"""

import os, pickle, lmdb, json, h5py, argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# =======================================================
# PARAMETRELER
# =======================================================
ESM2_H5_PATH   = os.path.join(PLANET_ROOT, "pdbbind2020_esm2_embeddings_v2.h5")
PCA_PARAMS_PATH = os.path.join(PLANET_ROOT, "esm2_pca_params.pkl")
PDB_MAP_JSON   = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json")
HEDEF_BOYUT    = 512
MAP_SIZE_GB    = 200

AA_3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V'
}

PCA_MEAN = None
PCA_TRANSFORM = None

# =======================================================
# YARDIMCI FONKSİYONLAR
# =======================================================

def load_pca_params(pca_path):
    global PCA_MEAN, PCA_TRANSFORM
    with open(pca_path, 'rb') as f:
        params = pickle.load(f)
    PCA_MEAN = params['mean']
    PCA_TRANSFORM = params['components']

def apply_pca(embedding):
    return np.dot(embedding - PCA_MEAN, PCA_TRANSFORM.T).astype(np.float32)

def parse_mapping_from_pdb(pdb_path):
    """PDB dosyasından (chain_id, res_num) → embedding index eşlemesi."""
    chain_res_info, chain_order, current_res = {}, [], None
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith("ATOM"): continue
                chain_id = line[21].strip() or 'A'
                res_num  = int(line[22:26].strip())
                res_type = line[17:20].strip()
                if res_type not in AA_3_TO_1: continue
                res_key = (chain_id, res_num)
                if chain_id not in chain_res_info:
                    chain_res_info[chain_id] = []
                    chain_order.append(chain_id)
                if res_key != current_res:
                    chain_res_info[chain_id].append(res_key)
                    current_res = res_key
        mapping, idx = {}, 0
        for c in chain_order:
            for r in chain_res_info[c]:
                mapping[r] = idx
                idx += 1
        return mapping
    except:
        return None

def inject_esm2_into_pocket(pocket_obj, mapping, esm2_embedding):
    """ComplexPocket objesine res_features ekler. Decoy'lara dokunmaz."""
    if not hasattr(pocket_obj, 'pocket_residues') or not pocket_obj.pocket_residues:
        return pocket_obj
    
    new_features = []
    for res_obj in pocket_obj.pocket_residues:
        chain_id = getattr(res_obj, 'chain_id', 'A')
        res_num  = res_obj.res_num
        key = (chain_id, res_num)
        if key in mapping:
            idx = mapping[key]
            if idx < esm2_embedding.shape[0]:
                res_obj.feature = esm2_embedding[idx]
                new_features.append(res_obj.feature)
    
    if new_features:
        pocket_obj.res_features = np.vstack(new_features).astype(np.float32)
    
    return pocket_obj

# =======================================================
# ANA FONKSİYON
# =======================================================

def update_set(src_dir, out_dir, set_name, h5_file, pdb_map):
    src_lmdb = os.path.join(src_dir, f'{set_name}.lmdb')
    src_meta = os.path.join(src_dir, f'{set_name}_meta.json')
    out_lmdb = os.path.join(out_dir, f'{set_name}.lmdb')
    out_meta = os.path.join(out_dir, f'{set_name}_meta.json')

    if not os.path.exists(src_lmdb):
        print(f'⚠️  {src_lmdb} bulunamadı, atlandı.')
        return

    print(f'\n{"="*60}')
    print(f'  {set_name.upper()} işleniyor...')

    with open(src_meta, 'r') as f:
        meta = json.load(f)

    # Kaynak LMDB
    src_env = lmdb.open(src_lmdb, readonly=True, lock=False,
                        readahead=False, meminit=False)

    # Hedef LMDB
    os.makedirs(out_lmdb, exist_ok=True)
    out_env = lmdb.open(out_lmdb, map_size=MAP_SIZE_GB * 1024**3,
                        subdir=True, meminit=False, map_async=True)

    errors, esm_added, esm_missing = 0, 0, 0
    new_meta = []

    with src_env.begin() as src_txn:
        with out_env.begin(write=True) as out_txn:
            for record in tqdm(meta, desc=f'  {set_name}'):
                pdb_id = record['pdb_id']
                key    = record['key']
                try:
                    # 1. Binder objesini oku
                    raw = src_txn.get(key.encode())
                    if raw is None:
                        errors += 1; continue
                    pocket_obj = pickle.loads(raw)

                    # 2. ESM2 embedding'i H5'ten oku
                    target = pdb_id if pdb_id in h5_file else \
                             pdb_id.upper() if pdb_id.upper() in h5_file else None

                    if target is not None:
                        esm2_emb = apply_pca(h5_file[target][:])

                        # 3. PDB mapping oluştur
                        pdb_path = pdb_map.get(pdb_id, {}).get('protein_pdb')
                        if pdb_path and os.path.exists(pdb_path):
                            mapping = parse_mapping_from_pdb(pdb_path)
                            if mapping:
                                # 4. Sadece binder objesine ekle (decoy listesine dokunma)
                                pocket_obj = inject_esm2_into_pocket(
                                    pocket_obj, mapping, esm2_emb)
                                esm_added += 1
                            else:
                                esm_missing += 1
                        else:
                            esm_missing += 1
                    else:
                        esm_missing += 1

                    # 5. Yeni LMDB'ye yaz
                    out_txn.put(
                        key.encode(),
                        pickle.dumps(pocket_obj, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                    new_meta.append(record)

                except Exception as e:
                    errors += 1
                    print(f'\n  ⚠️  Hata [{pdb_id}]: {e}')
                    continue

    src_env.close()
    out_env.sync()
    out_env.close()

    with open(out_meta, 'w') as f:
        json.dump(new_meta, f)

    print(f'  ✅ Kaydedildi : {out_lmdb}')
    print(f'  ESM2 eklendi  : {esm_added}')
    print(f'  ESM2 eksik    : {esm_missing}')
    print(f'  Hata          : {errors}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lmdb', required=True)
    parser.add_argument('--out_lmdb', required=True)
    parser.add_argument('--sets', nargs='+', default=['train', 'valid', 'core'])
    args = parser.parse_args()

    print('PCA parametreleri yükleniyor...')
    load_pca_params(PCA_PARAMS_PATH)

    print('PDB haritası yükleniyor...')
    with open(PDB_MAP_JSON, 'r') as f:
        pdb_map = json.load(f)

    os.makedirs(args.out_lmdb, exist_ok=True)

    print('ESM2 H5 dosyası açılıyor...')
    with h5py.File(ESM2_H5_PATH, 'r') as h5_file:
        for set_name in args.sets:
            update_set(args.src_lmdb, args.out_lmdb,
                       set_name, h5_file, pdb_map)

    print('\n🎉 Tüm setler tamamlandı!')
    print(f'Yeni LMDB: {args.out_lmdb}')


if __name__ == '__main__':
    main()