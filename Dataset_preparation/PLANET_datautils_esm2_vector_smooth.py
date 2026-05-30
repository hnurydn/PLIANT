import os, pickle, random, shutil, sys, argparse, json, getpass
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
# ESM2 uyumlu modülün
from chemutils_esm2_vector_smooth import tensorize_all 

# --- KESİN YOLLAR ---
PK_MAP_JSON = '/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json'
ESM_SUBDIR = "planet_esm2_pkl"

class ProLigDataset(Dataset):
    def __init__(self, data_pickle_path, batch_size, shuffle=True, decoy_flag=True, binder_per_batch=4):
        # 1. ⚡ YEREL SSD AYARI (HPC Uyumlu)
        self.local_ssd_dir = f"/tmp/{getpass.getuser()}_planet_cache"
        os.makedirs(self.local_ssd_dir, exist_ok=True)

        if not os.path.exists(data_pickle_path):
            print(f"⚠️ {data_pickle_path} bulunamadı!")
            self.data = []
            return

        with open(data_pickle_path, 'rb') as f:
            full_data = pickle.load(f)
        
        self.decoy_flag = decoy_flag

        # 2. ⚖️ DENGELEME MANTIĞI
        if shuffle and decoy_flag:
            binders = [r for r in full_data if float(r[1]) > 0.0]
            decoys = [r for r in full_data if float(r[1]) == 0.0]
            
            num_batches = len(binders) // binder_per_batch if binders else 1
            random.shuffle(binders)
            
            from itertools import cycle
            decoy_cycle = cycle(decoys) if decoys else []

            balanced_batches = []
            for i in range(num_batches):
                batch_binders = binders[i * binder_per_batch : (i + 1) * binder_per_batch]
                batch_decoys = [next(decoy_cycle) for _ in range(batch_size - binder_per_batch)] if decoys else []
                batch = batch_binders + batch_decoys
                random.shuffle(batch)
                balanced_batches.append(batch)
            self.data = balanced_batches
            print(f"✅ ESM2 DATASET HAZIR: {len(self.data)} Batch.")
        else:
            self.data = [full_data[i : i + batch_size] for i in range(0, len(full_data), batch_size)]

    def __len__(self): return len(self.data)

    def tensorize(self, idx):
        pocket_batch = []
        for record in self.data[idx]:
            source_path, pk_val = record[0], record[1]
            is_multi = record[2] if len(record) > 2 else False
            file_name = os.path.basename(source_path)
            
            # 3. 📂 ESM2 BYTE-OFFSET OPTIMIZASYONU
            if is_multi and "_esm512d.pkl" in source_path:
                byte_file_name = file_name.replace("_esm512d.pkl", "_byte_esm512d.pkl")
                potential_source = os.path.join(os.path.dirname(source_path), byte_file_name)
                if os.path.exists(potential_source):
                    file_name = byte_file_name
                    source_path = potential_source

            local_path = os.path.join(self.local_ssd_dir, file_name)

            # --- 🛡️ AKILLI DISK KONTROLÜ (10 GB EMNİYET KİLİDİ) ---
            if not os.path.exists(local_path):
                try:
                    _, _, free_space = shutil.disk_usage(self.local_ssd_dir)
                    if free_space > 10 * 1024 * 1024 * 1024: # 10GB Boş yer varsa kopyala
                        shutil.copy2(source_path, local_path)
                    else:
                        local_path = source_path # Yer yoksa Scratch'ten devam et
                except:
                    local_path = source_path
            # -----------------------------------------------------

            # 4. 📖 OKUMA VE OBJE KONTROLÜ
            try:
                with open(local_path, 'rb') as f:
                    content = pickle.load(f)
                
                if is_multi:
                    # Decoy listesinden seçim yap ve byte ise decode et
                    selected = random.choice(content) if isinstance(content, list) else content
                    obj = pickle.loads(selected) if isinstance(selected, bytes) else selected
                    
                    if hasattr(obj, 'pK'):
                        obj.pK = 0.0
                        pocket_batch.append(obj)
                else:
                    if hasattr(content, 'pK'):
                        pocket_batch.append(content)
            except:
                continue

        if not pocket_batch: return None
        
        # 5. 📦 VERİ PAKETLEME (Expected 3 Error Fix)
        # outputs: res_f, mol_f, mol_int, pl_int, pKs, pK_flags, labels
        outputs = tensorize_all(pocket_batch, True)
        
        # Eğitim scriptinin beklediği 3 ana grup: res, mol, targets
        res_f = outputs[0]
        mol_f = outputs[1]
        targets = (outputs[2], outputs[3], outputs[4], outputs[5], outputs[6])
        
        return res_f, mol_f, targets

    def __getitem__(self, idx):
        try: return self.tensorize(idx)
        except: return None

# --- PKL OLUŞTURUCU (MAIN) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index_dir', default="/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/index")
    args = parser.parse_args()

    with open(PK_MAP_JSON, 'r') as f:
        pk_data_map = json.load(f)
    
    sets = {'train': 'TrainSet_Strict_Cleaned.csv', 'valid': 'ValidSet.csv', 'core': 'CoreSet.csv'}

    for _type_, csv_name in sets.items():
        csv_path = os.path.join(args.index_dir, csv_name)
        if not os.path.exists(csv_path): continue

        df = pd.read_csv(csv_path)
        pdb_col = 'PDB_code' if 'PDB_code' in df.columns else 'pdb_id'
        dataset = []

        print(f"ESM2 İşleniyor: {csv_name}...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            pdb = str(row[pdb_col]).lower()
            if pdb not in pk_data_map: continue

            orig_path = pk_data_map[pdb]['protein_pdb']
            new_dir = os.path.dirname(orig_path).replace("pdbbind2020/", f"pdbbind2020/{ESM_SUBDIR}/")

            binder_path = os.path.join(new_dir, f"{pdb}_binder_esm512d.pkl")
            decoy_path = os.path.join(new_dir, f"{pdb}_decoys_esm512d.pkl")

            if os.path.exists(binder_path):
                dataset.append([binder_path, float(row['pK']), False])
                if _type_ != 'core' and os.path.exists(decoy_path):
                    dataset.append([decoy_path, 0.0, True])

        out_path = os.path.join(args.index_dir, f"{_type_}_esm2.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        print(f"✅ İndeks Kaydedildi: {out_path}")

if __name__ == "__main__":
    main()