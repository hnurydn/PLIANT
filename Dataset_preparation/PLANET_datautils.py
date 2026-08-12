import os, pickle, random, shutil, sys, argparse, json, getpass
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
# Kendi modüllerin (Aynı dizinde olduğu varsayılıyor)
from chemutils import ComplexPocket, tensorize_all
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# JSON dosyasının yolu
PK_MAP_PATH_LOCAL = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/PDBbind_structure_map_pkd_prepared.json")

class ProLigDataset(Dataset):
    def __init__(self, data_pickle_path, batch_size, shuffle=True, decoy_flag=True, binder_per_batch=4):
        # 1. ⚡ YEREL SSD AYARI: os.getlogin() yerine getpass.getuser() (HPC Uyumu)
        self.local_ssd_dir = f"/tmp/{getpass.getuser()}_planet_cache"
        os.makedirs(self.local_ssd_dir, exist_ok=True)
        
        if not os.path.exists(data_pickle_path):
            print(f"UYARI: {data_pickle_path} bulunamadı!")
            self.data = []
            return

        with open(data_pickle_path, 'rb') as f:
            full_data = pickle.load(f)
        
        self.decoy_flag = decoy_flag

        # 2. ⚖️ DENGELEME MANTIĞI: Binder ve Decoy dengeli batch'ler oluşturur
        if shuffle and decoy_flag:
            binders = [r for r in full_data if float(r[1]) > 0.0]
            decoys = [r for r in full_data if float(r[1]) == 0.0]
            
            if len(binders) == 0:
                self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]
            else:
                decoys_per_batch = max(1, batch_size - binder_per_batch)
                num_batches = max(1, len(binders) // binder_per_batch)
                
                random.shuffle(binders)
                random.shuffle(decoys)
                
                from itertools import cycle
                decoy_cycle = cycle(decoys) if decoys else []
                
                balanced_batches = []
                for i in range(num_batches):
                    batch = binders[i*binder_per_batch : (i+1)*binder_per_batch]
                    if decoy_cycle:
                        batch += [next(decoy_cycle) for _ in range(decoys_per_batch)]
                    random.shuffle(batch)
                    balanced_batches.append(batch)
                self.data = balanced_batches
                print(f"✅ DATASET YÜKLENDİ: {len(self.data)} dengeli batch hazır.")
        else:
            self.data = [full_data[i:i+batch_size] for i in range(0, len(full_data), batch_size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            return self.tensorize(idx)
        except Exception as e:
            # Hata durumunda batch'i atla ama logla
            # print(f"VERİ HATASI: {e}", file=sys.stderr)
            return None

    def tensorize(self, idx):
        pocket_batch = []
        
        for record in self.data[idx]:
            source_path = record[0] 
            is_multi = record[2] if len(record) > 2 else False
            file_name = os.path.basename(source_path)
            
            # 3. 📂 BYTE OPTIMIZASYONU VE SSD TRANSFERİ
            if is_multi and "decoys.pkl" in file_name:
                byte_file_name = file_name.replace("_decoys.pkl", "_decoys_byte.pkl")
                potential_source = os.path.join(os.path.dirname(source_path), byte_file_name)
                if os.path.exists(potential_source):
                    file_name = byte_file_name
                    source_path = potential_source

            local_path = os.path.join(self.local_ssd_dir, file_name)

            # --- 🛡️ AKILLI DISK KONTROLÜ (EN KRİTİK KISIM) ---
            if not os.path.exists(local_path):
                try:
                    # SSD'de en az 10 GB yer yoksa kopyalama yapma
                    _, _, free_space = shutil.disk_usage(self.local_ssd_dir)
                    if free_space > 10 * 1024 * 1024 * 1024: 
                        shutil.copy2(source_path, local_path)
                    else:
                        # Yer kalmadıysa Scratch üzerinden oku (Yavaş ama güvenli)
                        local_path = source_path
                except:
                    local_path = source_path 
            # -----------------------------------------------

            # 4. 📖 OKUMA VE OBJE KONTROLÜ
            try:
                with open(local_path, 'rb') as f:
                    content = pickle.load(f)
            except:
                continue

            if is_multi:
                if isinstance(content, list) and len(content) > 0:
                    selected = random.choice(content)
                    obj = pickle.loads(selected) if isinstance(selected, bytes) else selected
                    
                    # 'int' object has no attribute 'pK' hatasını önlemek için kontrol
                    if hasattr(obj, 'pK'):
                        obj.pK = 0.0
                        pocket_batch.append(obj)
            else:
                if hasattr(content, 'pK'):
                    pocket_batch.append(content)

        if not pocket_batch: return None

        # 5. 📦 VERİ PAKETLEME: return değerini tam olarak 3 parça yapar
        # outputs: res_f, mol_f, mol_int, pl_int, pKs, pK_flags, labels
        outputs = tensorize_all(pocket_batch, self.decoy_flag)
        
        res_f = outputs[0]
        mol_f = outputs[1]
        # Diğer 5 metrik targets paketine (tuple) gider
        targets = (outputs[2], outputs[3], outputs[4], outputs[5], outputs[6])
        
        return res_f, mol_f, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', required=True, help="PDBbind index dizini") 
    args = parser.parse_args()

    # Dosyaların gerçekte toplandığı yer
    ALL_PICKLES_DIR = os.path.join(PLANET_ROOT, "datasets/all_pickles")

    with open(PK_MAP_PATH_LOCAL, 'r') as f:
        pk_data_map = json.load(f)
    
    sets = {'train': 'TrainSet_Strict_Cleaned.csv', 'valid': 'ValidSet.csv', 'core': 'CoreSet.csv'}

    for _type_, csv_name in sets.items():
        csv_path = os.path.join(args.dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_name} bulunamadı, atlanıyor...")
            continue

        df = pd.read_csv(csv_path)
        pdb_col = 'PDB_code' if 'PDB_code' in df.columns else 'pdb_id'
        dataset = []

        print(f"📦 {_type_} seti işleniyor...")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pdb = str(row[pdb_col]).lower()
            
            # Alt klasör yollarını boşver, doğrudan all_pickles altına bak
            binder_path = os.path.join(ALL_PICKLES_DIR, f"{pdb}_binder.pkl")
            decoy_path = os.path.join(ALL_PICKLES_DIR, f"{pdb}_decoys.pkl")

            # Hata ayıklama için ilk satırı kontrol et
            if i == 0:
                print(f"🔍 İlk denenen yol: {binder_path}")

            if os.path.exists(binder_path):
                dataset.append([binder_path, float(row['pK']), False])
                # Core dahil her set için decoy ekle
                if os.path.exists(decoy_path):
                    dataset.append([decoy_path, 0.0, True])

        out_path = os.path.join(args.dir, f"{_type_}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        print(f"✅ İndeks oluşturuldu: {out_path} (Toplam: {len(dataset)} kayıt)")
        
if __name__ == "__main__":
    main()