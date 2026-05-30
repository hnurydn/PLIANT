import json, os, shutil
from tqdm import tqdm

JSON = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
DST  = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/all_pickles'
os.makedirs(DST, exist_ok=True)

with open(JSON) as f:
    data = json.load(f)

moved, missing, errors = 0, 0, 0

for pdb_id, info in tqdm(data.items(), desc='Taşınıyor'):
    src_dir = os.path.dirname(info['protein_pdb'])
    for suffix in ['_binder.pkl', '_decoys.pkl']:
        src = os.path.join(src_dir, pdb_id + suffix)
        dst = os.path.join(DST, pdb_id + suffix)
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                errors += 1
                print(f'Hata [{pdb_id}]: {e}')
        else:
            missing += 1

print(f'Taşınan: {moved} | Bulunamayan: {missing} | Hata: {errors}')