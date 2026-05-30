import pandas as pd
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import numpy as np
from rdkit import Chem
from collections import OrderedDict
import sys

# --- AYARLAR ---
POCKET_MAP_JSON = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
DECOY_CSV = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/decoy_candidates_find_500.csv"

PDB_ROOT_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/'
BANANA_POCKET_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/banana_pocket_pdbs_12A/'
BANANA_SCORES_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/banana_output_scores_filtered_2/'
TEMP_SMI_DIR = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/temp_smi_files/'
BANANA_DIR = '/arf/scratch/hnaydin/banana'

LIGAND_POCKET_CUTOFF = 12.0
CRITICAL_THRESHOLD = -3.0

RESIDUE_NAMES = {'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET', 'MSE'}

# --- GLOBAL WORKER DEĞİŞKENLERİ ---
_banana_model = None
_banana_cfg = None
_banana_device = "cpu"

def init_banana_worker():
    global _banana_model, _banana_cfg, _banana_device
    os.chdir(BANANA_DIR)
    sys.path.insert(0, BANANA_DIR)
    import torch
    from omegaconf import OmegaConf
    from terrace.comp_node import Input
    from terrace.batch import make_batch_td
    from datasets.data_types import IsActiveData
    from models.make_model import make_model
    _banana_cfg = OmegaConf.load(f"{BANANA_DIR}/configs/classification.yaml")
    in_node = Input(make_batch_td(IsActiveData.get_type_data(_banana_cfg)))
    _banana_model = make_model(_banana_cfg, in_node)
    _banana_model.load_state_dict(torch.load(f"{BANANA_DIR}/data/banana_final.pt", map_location='cpu'))
    _banana_model = _banana_model.to(_banana_device)
    _banana_model.eval()

def get_ligand_centroid(ligand_mol):
    if ligand_mol is None: return None
    try:
        conf = ligand_mol.GetConformer()
        coords = conf.GetPositions()
        return np.mean(coords, axis=0)
    except: return None

def parse_pdb_atom_line(line):
    try:
        return {
            'line': line,
            'residue_name': line[17:20].strip(),
            'chain_id': line[21],
            'residue_number': int(line[22:26]),
            'x': float(line[30:38]),
            'y': float(line[38:46]),
            'z': float(line[46:54])
        }
    except: return None

def create_pocket_pdb_from_data(pdb_id_data):
    pdb_id, data = pdb_id_data
    protein_pdb_path = data.get('protein_pdb')
    ligand_sdf_path = data.get('ligand_sdf')
    output_pdb_path = os.path.join(BANANA_POCKET_DIR, f"{pdb_id}_pocket.pdb")
    if os.path.exists(output_pdb_path):
        return pdb_id, output_pdb_path, "Already exists"
    if not protein_pdb_path or not os.path.exists(protein_pdb_path):
        return pdb_id, None, "Protein PDB not found"
    try:
        suppl = Chem.SDMolSupplier(ligand_sdf_path, removeHs=False, sanitize=True)
        ligand_mol = next(suppl, None)
        ligand_centroid = get_ligand_centroid(ligand_mol)
        if ligand_mol is None or ligand_centroid is None:
            return pdb_id, None, "Invalid ligand"
        pocket_residues_to_keep = set()
        residues_ordered = OrderedDict()
        with open(protein_pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_data = parse_pdb_atom_line(line)
                    if atom_data and atom_data['residue_name'] in RESIDUE_NAMES:
                        res_id = (atom_data['chain_id'], atom_data['residue_number'])
                        if res_id not in residues_ordered:
                            residues_ordered[res_id] = {'atoms': []}
                        residues_ordered[res_id]['atoms'].append(atom_data)
                        atom_coord = np.array([atom_data['x'], atom_data['y'], atom_data['z']])
                        if np.linalg.norm(atom_coord - ligand_centroid) <= LIGAND_POCKET_CUTOFF:
                            pocket_residues_to_keep.add(res_id)
        if not pocket_residues_to_keep:
            return pdb_id, None, "Pocket empty"
        pdb_lines = [a['line'].strip() for rid, rd in residues_ordered.items()
                     if rid in pocket_residues_to_keep for a in rd['atoms']]
        with open(output_pdb_path, 'w') as f:
            f.write('\n'.join(pdb_lines))
        return pdb_id, output_pdb_path, "Success"
    except Exception as e:
        return pdb_id, None, f"Error: {e}"

def run_inference_task(task_data):
    pdb_id, pocket_path, group_df = task_data
    if group_df is None or group_df.empty:
        return pdb_id, "Uyarı: Decoy adayı yok."
    final_csv_path = os.path.join(BANANA_SCORES_DIR, f"{pdb_id}_filtered.csv")
    try:
        import torch
        from terrace.batch import DataLoader
        from datasets.inference_dataset import InferenceDataset
        temp_smi_path = os.path.join(TEMP_SMI_DIR, f"{pdb_id}_candidates.smi")
        group_df[['SMILES', 'ID']].to_csv(temp_smi_path, sep='\t', header=False, index=False)
        dataset = InferenceDataset(_banana_cfg, temp_smi_path, pocket_path)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(_banana_device)
                output = _banana_model(batch).cpu().numpy()
                for x, out in zip(batch, output):
                    if not x.is_active:
                        out = -100
                    scores.append(float(out))
        if len(scores) != len(group_df):
            return pdb_id, f"HATA: Skor/ID uyuşmazlığı ({len(scores)} vs {len(group_df)})"
        result_df = pd.DataFrame({
            'PDB_ID': pdb_id,
            'CHEMBL_ID': group_df['ID'].tolist(),
            'BANANA_P_binder': scores
        })
        final_filtered_df = result_df[result_df['BANANA_P_binder'] < CRITICAL_THRESHOLD]
        final_filtered_df.to_csv(final_csv_path, index=False)
        return pdb_id, f"Başarılı: {len(final_filtered_df)} inaktif kaydedildi."
    except Exception as e:
        return pdb_id, f"HATA: {e}"

def main_pipeline(n_jobs):
    print(f"--- 6. CEP OLUŞTURMA VE BANANA SKORLAMA ({n_jobs} çekirdek) ---")
    os.makedirs(BANANA_POCKET_DIR, exist_ok=True)
    os.makedirs(BANANA_SCORES_DIR, exist_ok=True)
    os.makedirs(TEMP_SMI_DIR, exist_ok=True)
    with open(POCKET_MAP_JSON, 'r') as f:
        pocket_map = json.load(f)
    with Pool(processes=n_jobs) as pool:
        pocket_results = list(tqdm(
            pool.imap_unordered(create_pocket_pdb_from_data, list(pocket_map.items())),
            total=len(pocket_map), desc="Cep Hazırlama"
        ))
    valid_pockets = {pid: path for pid, path, status in pocket_results
                     if path and status in ["Success", "Already exists"]}
    print("\nDecoy CSV okunuyor...")
    df_decoys = pd.read_csv(DECOY_CSV).rename(columns={'chembl_smiles': 'SMILES', 'chembl_id': 'ID'})
    grouped_data = {k: v for k, v in df_decoys.groupby('binder_pdb_id')}
    del df_decoys
    target_pdb_ids = set(grouped_data.keys())
    valid_pockets = {pid: path for pid, path in valid_pockets.items() if pid in target_pdb_ids}
    print(f"Hedef PDB sayısı: {len(valid_pockets)}")
    tasks_score = []
    for pid, path in valid_pockets.items():
        if os.path.exists(os.path.join(BANANA_SCORES_DIR, f"{pid}_filtered.csv")):
            continue
        tasks_score.append((pid, path, grouped_data.get(pid)))
    if not tasks_score:
        print("Tüm skorlamalar zaten tamamlanmış.")
        return
    print(f"\n[SKORLAMA] {len(tasks_score)} görev başlatılıyor...")
    with Pool(processes=n_jobs, initializer=init_banana_worker) as pool:
        score_results = list(tqdm(
            pool.imap_unordered(run_inference_task, tasks_score),
            total=len(tasks_score), desc="BANANA Skorlama"
        ))
    print("\n--- İŞLEM TAMAMLANDI ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--njobs', type=int, default=16)
    args = parser.parse_args()
    main_pipeline(n_jobs=args.njobs)
