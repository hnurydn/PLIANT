import os
import pickle
import numpy as np
import h5py
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
import sys
from sklearn.decomposition import PCA 
import glob 

# =======================================================
# PARAMETRELER
# =======================================================
ESM2_H5_PATH = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/pdbbind2020_esm2_embeddings_v2.h5' 
PCA_PARAMS_PATH = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/esm2_pca_params.pkl' 
HEDEF_BOYUT = 512

PDBBIND_KOK_DIR = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020"
PDB_MAP_JSON = os.path.join(PDBBIND_KOK_DIR, "PDBbind_structure_map_pkd_prepared.json")

ALL_PICKLES_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/all_pickles'
ESM_OUTPUT_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/all_pickles_esm'

N_JOBS = 40 

PCA_MEAN = None
PCA_TRANSFORM = None

AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

# =======================================================
# YARDIMCI FONKSİYONLAR
# =======================================================

def calculate_pca_whitening_params(h5_path, output_path, target_dim):
    global PCA_MEAN, PCA_TRANSFORM
    print("HDF5 dosyasındaki tüm ESM-2 gömülmeleri okunuyor...")
    all_embeddings = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for pdb_id in tqdm(list(f.keys()), desc="Gömülmeleri oku"):
                all_embeddings.append(f[pdb_id][()])
    except Exception as e:
        print(f"Hata: {e}"); return False

    all_embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    pca = PCA(n_components=target_dim, whiten=True)
    pca.fit(all_embeddings_matrix)
    PCA_MEAN = pca.mean_.astype(np.float32)
    PCA_TRANSFORM = pca.components_.astype(np.float32)
    
    with open(output_path, 'wb') as f:
        pickle.dump({'mean': PCA_MEAN, 'components': PCA_TRANSFORM, 'target_dim': target_dim}, f)
    return True

def load_or_calculate_pca_params(h5_path, pca_path, target_dim):
    if os.path.exists(pca_path):
        global PCA_MEAN, PCA_TRANSFORM
        with open(pca_path, 'rb') as f:
            params = pickle.load(f)
            PCA_MEAN, PCA_TRANSFORM = params['mean'], params['components']
        return True
    return calculate_pca_whitening_params(h5_path, pca_path, target_dim)

def apply_pca_transform(full_embedding):
    if PCA_TRANSFORM is None: return full_embedding.astype(np.float32)
    return np.dot(full_embedding - PCA_MEAN, PCA_TRANSFORM.T)

def parse_fasta_from_pdb(pdb_file_path):
    chain_res_info, chain_order, current_res_num = {}, [], None
    try:
        with open(pdb_file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain_id = line[21].strip() or 'A'
                    res_num = int(line[22:26].strip())
                    res_type = line[17:20].strip()
                    if res_type not in AA_3_TO_1: continue
                    res_key = (chain_id, res_num)
                    if chain_id not in chain_res_info:
                        chain_res_info[chain_id] = []
                        chain_order.append(chain_id)
                    if res_key != current_res_num:
                        chain_res_info[chain_id].append(res_key)
                        current_res_num = res_key
        mapping, idx = {}, 0
        for c in chain_order:
            for r in chain_res_info[c]:
                mapping[r], idx = idx, idx + 1
        return mapping
    except: return None

def update_molecule_features(mol_obj, mapping, esm2_embedding):
    if hasattr(mol_obj, 'pocket_residues') and mol_obj.pocket_residues:
        new_feature_list = []
        for res_obj in mol_obj.pocket_residues:
            chain_id = getattr(res_obj, 'chain_id', 'A')
            res_num = res_obj.res_num
            key = (chain_id, res_num)
            if key in mapping:
                res_idx = mapping[key]
                if res_idx < esm2_embedding.shape[0]:
                    res_obj.feature = esm2_embedding[res_idx].astype(np.float32)
                    new_feature_list.append(res_obj.feature)
        if new_feature_list:
            mol_obj.res_features = np.vstack(new_feature_list)
    return mol_obj

# =======================================================
# PKL GÜNCELLEME FONKSİYONU
# =======================================================

def process_single_pkl_update(task_data, esm2_h5_file_path, lock):
    pdb_id, data = task_data

    # ADIM A: Embedding Oku
    esm2_full_embedding = None
    with lock:
        with h5py.File(esm2_h5_file_path, 'r') as h5f:
            target = pdb_id if pdb_id in h5f else (pdb_id.upper() if pdb_id.upper() in h5f else None)
            if target: esm2_full_embedding = apply_pca_transform(h5f[target][:])

    if esm2_full_embedding is None:
        return (pdb_id, "HATA: Embedding yok")

    # ADIM B: Mapping Oluştur
    pdb_path = data.get('protein_pdb')
    mapping = parse_fasta_from_pdb(pdb_path)
    if not mapping:
        return (pdb_id, "HATA: Mapping yapilamadi")

    # ADIM C: all_pickles altındaki binder ve decoy pkl'leri işle
    os.makedirs(ESM_OUTPUT_DIR, exist_ok=True)
    pkl_files = glob.glob(os.path.join(ALL_PICKLES_DIR, f"{pdb_id}_*.pkl"))

    if not pkl_files:
        return (pdb_id, "UYARI: PKL bulunamadı")

    updated_count = 0
    for p in pkl_files:
        out_p = os.path.join(ESM_OUTPUT_DIR, 
                             os.path.basename(p).replace(".pkl", f"_esm{HEDEF_BOYUT}d.pkl"))
        
        # Zaten varsa atla (üzerine yazma)
        if os.path.exists(out_p):
            updated_count += 1
            continue

        try:
            with open(p, 'rb') as f:
                pkl_obj = pickle.load(f)

            if isinstance(pkl_obj, list):  # decoys.pkl
                for mol in pkl_obj:
                    update_molecule_features(mol, mapping, esm2_full_embedding)
            else:  # binder.pkl
                update_molecule_features(pkl_obj, mapping, esm2_full_embedding)

            with open(out_p, 'wb') as f:
                pickle.dump(pkl_obj, f, pickle.HIGHEST_PROTOCOL)
            updated_count += 1
        except:
            continue

    return (pdb_id, f"✅ Basarili ({updated_count} PKL)")

# =======================================================
# MAIN
# =======================================================

def main():
    print("--- ESM-2 PKL GÜNCELLEYİCİ BAŞLATILDI ---")
    if not load_or_calculate_pca_params(ESM2_H5_PATH, PCA_PARAMS_PATH, HEDEF_BOYUT):
        sys.exit(1)

    with open(PDB_MAP_JSON, 'r') as f:
        pocket_map = json.load(f)

    # Sadece all_pickles'ta binder.pkl olan PDB'leri işle
    available_pdb_ids = set(
        f.replace('_binder.pkl', '')
        for f in os.listdir(ALL_PICKLES_DIR)
        if f.endswith('_binder.pkl')
    )
    tasks = [(pid, data) for pid, data in pocket_map.items() if pid in available_pdb_ids]
    print(f"İşlenecek PDB sayısı: {len(tasks)}")

    m = Manager()
    lock = m.Lock()

    task_processor = partial(process_single_pkl_update,
                             esm2_h5_file_path=ESM2_H5_PATH,
                             lock=lock)

    with Pool(processes=N_JOBS) as pool:
        for (pid, status) in tqdm(pool.imap_unordered(task_processor, tasks), total=len(tasks)):
            if not status.startswith("✅"):
                tqdm.write(f"[{pid}] {status}")

    print("\n--- İŞLEM TAMAMLANDI ---")

if __name__ == "__main__":
    main()