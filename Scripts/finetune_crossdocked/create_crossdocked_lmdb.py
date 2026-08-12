"""
create_crossdocked_lmdb.py
===========================
Builds an LMDB from CrossDocked PKL files, adding ESM-2 embeddings.

Each binder is written as its own LMDB record.
The meta JSON has a pocket_name field -- the DataLoader groups by pocket.

Output:
  lmdb_crossdocked/
    train.lmdb
    train_meta.json

Meta JSON structure:
  [
    {
      "key":         "1433B_HUMAN_1_240_pep_0__2bq0_5f74_amp",
      "pocket_name": "1433B_HUMAN_1_240_pep_0",
      "pdb_id":      "2bq0",
      "pK":          5.1192
    },
    ...
  ]

Usage:
    python create_crossdocked_lmdb.py \
        --pkl_dir      $PLANET_ROOT/.../crossdocked_pkls \
        --json_path    $PLANET_ROOT/.../crossdocked_new_records.json \
        --h5_path      $PLANET_ROOT/.../pdbbind2020_esm2_embeddings_v2.h5 \
        --pca_path     $PLANET_ROOT/.../esm2_pca_params.pkl \
        --extracted_dir $PLANET_ROOT/.../crossdocked_extracted \
        --out_lmdb     $PLANET_ROOT/.../lmdb_crossdocked \
        --map_size_gb  100
"""

import os, sys, json, pickle, argparse, glob
import numpy as np
import lmdb, h5py
from tqdm import tqdm

# ── PCA ──────────────────────────────────────────────────────────────────────
PCA_MEAN      = None
PCA_TRANSFORM = None

def load_pca_params(pca_path):
    global PCA_MEAN, PCA_TRANSFORM
    with open(pca_path, 'rb') as f:
        params = pickle.load(f)
    PCA_MEAN      = params['mean']
    PCA_TRANSFORM = params['components']
    print(f"✅ PCA yüklendi: {PCA_TRANSFORM.shape}")

def apply_pca(embedding):
    return np.dot(embedding - PCA_MEAN, PCA_TRANSFORM.T).astype(np.float32)

# ── PDB → mapping ─────────────────────────────────────────────────────────────
AA_3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V'
}

def parse_mapping_from_pdb(pdb_path):
    """(chain_id, res_num) → embedding index eşlemesi."""
    chain_res, chain_order, current_res = {}, [], None
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith("ATOM"): continue
                chain_id = line[21].strip() or 'A'
                res_num  = int(line[22:26].strip())
                res_type = line[17:20].strip()
                if res_type not in AA_3_TO_1: continue
                res_key = (chain_id, res_num)
                if chain_id not in chain_res:
                    chain_res[chain_id] = []
                    chain_order.append(chain_id)
                if res_key != current_res:
                    chain_res[chain_id].append(res_key)
                    current_res = res_key
        mapping, idx = {}, 0
        for c in chain_order:
            for r in chain_res[c]:
                mapping[r] = idx
                idx += 1
        return mapping
    except:
        return None

def inject_esm2(pocket_obj, mapping, esm2_embedding):
    """ComplexPocket'a ESM2 res_features ekler."""
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

# ── Receptor PDB bul ──────────────────────────────────────────────────────────
def find_receptor_pdb(extracted_dir, folder, pdb_id):
    folder_path = os.path.join(extracted_dir, folder)
    pdb_file = os.path.join(folder_path, f'{pdb_id}_A_rec.pdb')
    if os.path.exists(pdb_file):
        return pdb_file
    matches = glob.glob(os.path.join(folder_path, f'{pdb_id}_*.pdb'))
    return matches[0] if matches else None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir',       required=True,
                        help='crossdocked_pkls klasörü')
    parser.add_argument('--json_path',     required=True,
                        help='crossdocked_new_records.json')
    parser.add_argument('--h5_path',       required=True,
                        help='ESM2 H5 dosyası')
    parser.add_argument('--pca_path',      required=True,
                        help='esm2_pca_params.pkl')
    parser.add_argument('--extracted_dir', required=True,
                        help='crossdocked_extracted klasörü')
    parser.add_argument('--out_lmdb',      required=True,
                        help='Çıktı LMDB klasörü')
    parser.add_argument('--map_size_gb',   type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_lmdb, exist_ok=True)
    out_lmdb_path = os.path.join(args.out_lmdb, 'train.lmdb')
    out_meta_path = os.path.join(args.out_lmdb, 'train_meta.json')

    # PCA yükle
    load_pca_params(args.pca_path)

    # JSON kayıtları yükle ve best-pose seç
    # Her pocket-ligand ikilisi için: tt_min varsa onu al, yoksa tt_docked
    raw_records = json.load(open(args.json_path))
    print(f"Ham JSON kaydı: {len(raw_records)}")

    def get_lig_code(lig_file):
        base = lig_file.replace('.gninatypes', '')
        if 'tt_min' in base:
            core = base.split('_tt_min')[0].replace('_lig', '')
        else:
            core = base.split('_tt_docked')[0].replace('_lig', '')
        if '_rec_' in core:
            return core.split('_rec_')[1]
        return None

    best = {}
    for r in raw_records:
        folder   = r['ligand'].split('/')[0]
        lig_file = r['ligand'].split('/')[-1]
        lig_code = get_lig_code(lig_file)
        if not lig_code:
            continue
        key  = f"{folder}__{r['pdb_id']}_{lig_code}"
        pose = 'min' if 'tt_min' in lig_file else 'docked'
        if key not in best or pose == 'min':
            best[key] = r

    records = list(best.values())
    tt_min_c    = sum(1 for r in records if 'tt_min'    in r['ligand'])
    tt_docked_c = sum(1 for r in records if 'tt_docked' in r['ligand'])
    print(f"Best-pose seçilen: {len(records)}")
    print(f"  tt_min (crystal): {tt_min_c}")
    print(f"  tt_docked       : {tt_docked_c}")

    def get_pkl_key(lig_file):
        return get_lig_code(lig_file)

    # LMDB aç
    env = lmdb.open(
        out_lmdb_path, subdir=True,
        map_size=args.map_size_gb * 1024**3,
        meminit=False, map_async=True
    )

    meta       = []
    written    = 0
    esm_added  = 0
    esm_miss   = 0
    pkl_miss   = 0
    errors     = 0

    print("\nLMDB oluşturuluyor...")

    with h5py.File(args.h5_path, 'r') as h5f:
        with env.begin(write=True) as txn:
            for r in tqdm(records, desc="Kayıt işleniyor"):
                pdb_id      = r['pdb_id']
                pk          = r.get('pk', 0.0)
                folder      = r['ligand'].split('/')[0]
                lig_file    = r['ligand'].split('/')[-1]

                # PKL key
                lig_code = get_pkl_key(lig_file)
                if lig_code is None:
                    errors += 1; continue

                pkl_path = os.path.join(
                    args.pkl_dir, folder,
                    f'{pdb_id}_{lig_code}_binder.pkl'
                )
                if not os.path.exists(pkl_path):
                    pkl_miss += 1; continue

                try:
                    with open(pkl_path, 'rb') as f:
                        pocket_obj = pickle.load(f)
                except Exception as e:
                    errors += 1; continue

                # ESM2 embedding ekle
                h5_key = pdb_id.upper()
                if h5_key in h5f:
                    esm2_emb = apply_pca(h5f[h5_key][:])
                    rec_pdb  = find_receptor_pdb(args.extracted_dir, folder, pdb_id)
                    if rec_pdb:
                        mapping = parse_mapping_from_pdb(rec_pdb)
                        if mapping:
                            pocket_obj = inject_esm2(pocket_obj, mapping, esm2_emb)
                            esm_added += 1
                        else:
                            esm_miss += 1
                    else:
                        esm_miss += 1
                else:
                    esm_miss += 1

                # LMDB key: pocket_name__pdb_id_lig_code
                lmdb_key = f"{folder}__{pdb_id}_{lig_code}"

                txn.put(
                    lmdb_key.encode(),
                    pickle.dumps(pocket_obj, protocol=pickle.HIGHEST_PROTOCOL)
                )

                # Meta kaydı
                meta.append({
                    'key':         lmdb_key,
                    'pocket_name': folder,
                    'pdb_id':      pdb_id,
                    'pK':          pk,
                })
                written += 1

    env.sync()
    env.close()

    # Meta JSON kaydet
    with open(out_meta_path, 'w') as f:
        json.dump(meta, f)

    print(f"\n{'='*50}")
    print(f"✅ Yazılan kayıt  : {written}")
    print(f"ESM2 eklendi     : {esm_added}")
    print(f"ESM2 eksik       : {esm_miss}")
    print(f"PKL bulunamadı   : {pkl_miss}")
    print(f"Hata             : {errors}")
    print(f"\nLMDB : {out_lmdb_path}")
    print(f"Meta : {out_meta_path}")

    # Pocket istatistikleri
    from collections import Counter
    pocket_counts = Counter(m['pocket_name'] for m in meta)
    print(f"\nUnique pocket    : {len(pocket_counts)}")
    print(f"Ort. binder/pocket: {sum(pocket_counts.values())/len(pocket_counts):.1f}")
    print(f"Min binder/pocket: {min(pocket_counts.values())}")
    print(f"Max binder/pocket: {max(pocket_counts.values())}")


if __name__ == '__main__':
    main()