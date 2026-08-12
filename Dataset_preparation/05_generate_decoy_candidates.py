# 05_generate_decoy_candidates.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import os
import multiprocessing
import pickle
import time
import random
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- AYARLAR ---
# Genel Ana Dizin (BASE_DIR'in kendisi)
BASE_DIR = os.path.join(PLANET_ROOT, "datasets")

# Girdiler
CHEMBL_SDF = os.path.join(BASE_DIR, "chembl_29.sdf")
# BINDER_PROPS_CSV = os.path.join(BASE_DIR, "pdbbind2020", "binder_properties.csv")

# Çıktılar
OUTPUT_PKL = os.path.join(BASE_DIR, "chembl_29_full_data.pkl")
# OUTPUT_CANDIDATES_CSV = os.path.join(BASE_DIR, "decoy_candidates_find_1000_suitable.csv") 
BINDER_PROPS_CSV = os.path.join(BASE_DIR, "pdbbind2020", "binder_properties_empty_scores.csv")
OUTPUT_CANDIDATES_CSV = os.path.join(BASE_DIR, "decoy_candidates_kalanlar_2.csv")

# Parametreler
DECOY_LIMIT = 100       # Her PDB için maksimum decoy adayı
MIN_MATCH_SCORE = 5     # 8 kriterden en az 6'sını sağlamalı


# ==============================================================================
# AŞAMA A: CHEMBL ÖN HESAPLAMA
# ==============================================================================

g_chembl_path = ""

def init_worker(chembl_sdf_path):
    """Her worker'a ChEMBL SDF dosyasının yolunu öğretir."""
    global g_chembl_path
    g_chembl_path = chembl_sdf_path

def get_properties_smiles_id(mol):
    """Bir molekülden ID, SMILES ve 8 fizikokimyasal özelliği hesaplar."""
    if mol is None:
        return None
    try:
        props = {
            'MolWt':            Descriptors.MolWt(mol),
            'LogP':             Descriptors.MolLogP(mol),
            'NumHDonors':       Descriptors.NumHDonors(mol),
            'NumHAcceptors':    Descriptors.NumHAcceptors(mol),
            'NumChiralCenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'NumRings':         Descriptors.RingCount(mol),
            'NumHeavyAtoms':    mol.GetNumHeavyAtoms(),
            'NumTotalAtoms':    mol.GetNumAtoms()
        }
        chembl_id = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unknown_ID'
        smiles = Chem.MolToSmiles(mol)
        return {'id': chembl_id, 'smiles': smiles, 'props': props}
    except Exception:
        return None

def read_one_molecule_block(f, start_offset):
    """SDF dosyasından tek bir molekül bloğunu okur."""
    mol_byte_block = b""
    try:
        f.seek(start_offset)
        for line in f:
            mol_byte_block += line
            if line.strip() == b'$$$$':
                break
        return mol_byte_block.decode('utf-8', errors='ignore')
    except Exception:
        return None

def process_chunk(offset_list):
    """Bir grup byte-offset'e karşılık gelen molekülleri işler."""
    global g_chembl_path
    results = []
    try:
        with open(g_chembl_path, 'rb') as f:
            for offset in offset_list:
                mol_string = read_one_molecule_block(f, offset)
                if mol_string:
                    suppl = Chem.SDMolSupplier()
                    suppl.SetData(mol_string, removeHs=False, sanitize=True)
                    mol = next(suppl, None)
                    if mol:
                        mol_data = get_properties_smiles_id(mol)
                        if mol_data:
                            mol_data['offset'] = offset
                            results.append(mol_data)
    except Exception:
        return []
    return results

def create_chembl_index(chembl_sdf):
    """SDF dosyasını baştan sona tarayarak her molekülün byte-offset'ini çıkarır."""
    print("ChEMBL SDF dosyası taranarak byte-offset indeksi oluşturuluyor...")
    molecule_index_list = []
    try:
        total_size = os.path.getsize(chembl_sdf)
        with open(chembl_sdf, 'rb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="İndeks Oluşturuluyor") as pbar:
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                pbar.update(len(line_bytes))
                if line_bytes.strip() == b'$$$$':
                    next_mol_start_offset = f.tell()
                    temp_pos = f.tell()
                    next_line_check = f.readline()
                    f.seek(temp_pos)
                    if next_line_check and next_line_check.strip() != b'$$$$':
                        molecule_index_list.append(next_mol_start_offset)
    except FileNotFoundError:
        print(f"HATA: SDF dosyası bulunamadı: {chembl_sdf}")
        return []
    except Exception as e:
        print(f"İndeks oluşturulurken hata: {e}")
        return []
    print(f"Toplam {len(molecule_index_list)} molekül konumu indekslendi.")
    return molecule_index_list

def run_precomputation(chembl_sdf, output_pkl, num_cpus):
    """
    ChEMBL SDF dosyasındaki tüm moleküllerin özelliklerini hesaplar ve PKL'ye kaydeder.
    PKL zaten mevcutsa doğrudan yükler.
    """
    print("\n--- AŞAMA A: ChEMBL Özelliklerini PKL Olarak Ön Hesaplama ---")

    if os.path.exists(output_pkl):
        print(f"PKL dosyası zaten mevcut, yükleniyor: {output_pkl}")
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)

    molecule_index_list = create_chembl_index(chembl_sdf)
    if not molecule_index_list:
        return []

    total_mols = len(molecule_index_list)
    num_chunks_per_cpu = 8
    chunksize = max(1, total_mols // (num_cpus * num_chunks_per_cpu))
    total_chunks = (total_mols + chunksize - 1) // chunksize
    print(f"{num_cpus} çekirdek, {chunksize} boyutunda {total_chunks} chunk kullanılacak.")

    offset_chunks = (molecule_index_list[i:i + chunksize] for i in range(0, total_mols, chunksize))
    all_results = []

    with multiprocessing.Pool(processes=num_cpus, initializer=init_worker, initargs=(chembl_sdf,)) as pool:
        for result_chunk in tqdm(pool.imap_unordered(process_chunk, offset_chunks, chunksize=1),
                                 total=total_chunks, desc="Özellikler Hesaplanıyor"):
            if result_chunk:
                all_results.extend(result_chunk)

    print(f"\nToplam {len(all_results)} molekülün verisi hesaplandı.")
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"PKL kaydedildi: {output_pkl}")
    return all_results


# ==============================================================================
# AŞAMA B: DECOY SEÇİMİ
# ==============================================================================

g_chembl_data_list = []
g_chembl_list_size = 0

def init_selection_worker(chembl_full_data_list):
    """Her worker'a ChEMBL veri listesini yükler."""
    global g_chembl_data_list, g_chembl_list_size
    g_chembl_data_list = chembl_full_data_list
    g_chembl_list_size = len(g_chembl_data_list)

def check_criteria(chembl_props, binder_row):
    """
    8 fizikokimyasal kriteri kontrol eder.
    Kaç tanesini sağladığını (0-8 arası) döndürür.
    """
    score = 0
    try:
        b_molwt  = float(binder_row['MolWt'])
        b_logp   = float(binder_row['LogP'])
        b_hdon   = float(binder_row['NumHDonors'])
        b_hacc   = float(binder_row['NumHAcceptors'])
        b_chiral = float(binder_row['NumChiralCenters'])
        b_rings  = float(binder_row['NumRings'])
        b_heavy  = float(binder_row['NumHeavyAtoms'])
        b_total  = float(binder_row['NumTotalAtoms'])

        '''if abs(chembl_props['MolWt']            - b_molwt)  < (b_molwt * 0.15): score += 1
        if abs(chembl_props['LogP']             - b_logp)   < 1.5:             score += 1
        if abs(chembl_props['NumHDonors']       - b_hdon)   <= 1:              score += 1
        if abs(chembl_props['NumHAcceptors']    - b_hacc)   <= 1:              score += 1
        if abs(chembl_props['NumChiralCenters'] - b_chiral) <= 1:              score += 1
        if abs(chembl_props['NumRings']         - b_rings)  <= 1:              score += 1
        if abs(chembl_props['NumHeavyAtoms']    - b_heavy)  <= 2:              score += 1
        if abs(chembl_props['NumTotalAtoms']    - b_total)  < (b_total * 0.15): score += 1'''
        
        if abs(chembl_props['MolWt']            - b_molwt)  < (b_molwt * 0.20): score += 1
        if abs(chembl_props['LogP']             - b_logp)   < 2.0:               score += 1
        if abs(chembl_props['NumHDonors']       - b_hdon)   <= 2:                score += 1
        if abs(chembl_props['NumHAcceptors']    - b_hacc)   <= 2:                score += 1
        if abs(chembl_props['NumChiralCenters'] - b_chiral) <= 2:                score += 1
        if abs(chembl_props['NumRings']         - b_rings)  <= 2:                score += 1
        if abs(chembl_props['NumHeavyAtoms']    - b_heavy)  <= 3:                score += 1
        if abs(chembl_props['NumTotalAtoms']    - b_total)  < (b_total * 0.20):  score += 1
    except Exception:
        return 0
    return score

def process_ONE_binder(binder_row):
    """
    Tek bir binder (PDB) için DECOY_LIMIT kadar uygun decoy adayı arar.

    Çalışma mantığı:
    1. ChEMBL listesinde rastgele bir başlangıç noktası seçilir.
       (Her PDB farklı bir noktadan başlar → çeşitlilik sağlanır)
    2. Oradan itibaren döngüsel olarak (% operatörü) tüm liste taranır.
       (Başlangıç noktasından sona gidilir, sonra başa dönülür,
        başlangıç noktasına kadar gelinir → hiçbir molekül atlanmaz)
    3. Kritere uyan her molekül eklenir, DECOY_LIMIT'e ulaşınca durulur.
    4. Tüm liste tarandıktan sonra hâlâ DECOY_LIMIT'e ulaşılamamışsa
       bulunan kadarıyla devam edilir ve uyarı verilir.
    """
    global g_chembl_data_list, g_chembl_list_size

    binder_pdb_id = binder_row['pdb_id']
    matching_candidates = []
    found_count = 0

    if g_chembl_list_size == 0:
        return []

    # Her PDB için farklı bir rastgele başlangıç noktası
    start_index = random.randrange(g_chembl_list_size)

    for i in range(g_chembl_list_size):
        # Döngüsel indeks: sona gelince başa döner
        idx = (start_index + i) % g_chembl_list_size
        entry = g_chembl_data_list[idx]

        if not entry or 'props' not in entry:
            continue

        if check_criteria(entry['props'], binder_row) >= MIN_MATCH_SCORE:
            chembl_smiles = entry.get('smiles', '')
            if chembl_smiles:
                chembl_id = entry.get('id', 'Unknown_ID')
                matching_candidates.append(f"{binder_pdb_id},{chembl_id},{chembl_smiles}")
                found_count += 1

                if found_count >= DECOY_LIMIT:
                    break  # Yeterli aday bulundu, dur

    if found_count < DECOY_LIMIT:
        print(f"UYARI: {binder_pdb_id} için tüm ChEMBL tarandı, "
              f"sadece {found_count}/{DECOY_LIMIT} aday bulunabildi.")

    return matching_candidates


def run_selection(binder_props_csv, chembl_full_data_list, output_candidates_csv, num_cpus):
    """
    Tüm binder'lar için decoy seçimini yönetir.
    Daha önce işlenmiş PDB'leri atlayarak kaldığı yerden devam edebilir.
    """
    if not chembl_full_data_list:
        print("KRİTİK HATA: ChEMBL verisi bellekte mevcut değil.")
        return

    try:
        df_binders = pd.read_csv(binder_props_csv)
        binder_records = df_binders.to_dict('records')
    except Exception as e:
        print(f"KRİTİK HATA: Binder CSV okunamadı: {e}")
        return

    if not binder_records:
        print("UYARI: Binder listesi boş!")
        return

    # --- 1. Daha önce işlenmiş PDB'leri bul ---
    completed_pdb_ids = set()

    if os.path.exists(output_candidates_csv):
        try:
            df_existing = pd.read_csv(output_candidates_csv, usecols=['binder_pdb_id'])
            if not df_existing.empty:
                counts = df_existing.groupby('binder_pdb_id').size()
                completed_pdb_ids = set(counts[counts >= 20].index)
                print(f'Bilgi: {len(completed_pdb_ids)} PDB tamamlanmis, atlanacak.')
                print(f'Bilgi: {len(counts[counts < 20])} PDB eksik, tekrar islenecek.')
        except Exception as e:
            print(f"Uyarı: Mevcut CSV okunamadı ({e}), sıfırdan başlanıyor.")

    # --- 2. Sadece işlenmemiş PDB'leri al ---
    binder_records = [r for r in binder_records if r['pdb_id'] not in completed_pdb_ids]
    total_to_process = len(binder_records)
    print(f"İşlenecek PDB sayısı: {total_to_process}")

    if total_to_process == 0:
        print("Tüm PDB'ler tamamlanmış. İşlem sonlandırıldı.")
        return

    # --- 3. Seçimi başlat ---
    with open(output_candidates_csv, 'a') as f_out:
        # Dosya yeni oluşturuluyorsa başlık yaz
        if os.path.getsize(output_candidates_csv) == 0:
            f_out.write("binder_pdb_id,chembl_id,chembl_smiles\n")

        with multiprocessing.Pool(
            processes=num_cpus,
            initializer=init_selection_worker,
            initargs=(chembl_full_data_list,)
        ) as pool:
            results_iterator = pool.imap_unordered(process_ONE_binder, binder_records, chunksize=1)
            newly_found = 0

            for candidate_lines in tqdm(results_iterator, total=total_to_process, desc="Binder'lar İşleniyor"):
                if candidate_lines:
                    newly_found += len(candidate_lines)
                    for line in candidate_lines:
                        f_out.write(line + "\n")
                    f_out.flush()  # Anlık diske yaz (veri kaybını önler)

    print(f"\n--- Tamamlandı ---")
    print(f"Yeni eklenen aday sayısı: {newly_found}")


# ==============================================================================
# ANA ÇALIŞTIRMA
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()

    try:
        num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', os.cpu_count()))
    except ValueError:
        num_cpus = 1

    print(f"Kullanılacak CPU sayısı: {num_cpus}")

    # Aşama A: ChEMBL özelliklerini hesapla / PKL'den yükle
    chembl_full_data_list = run_precomputation(CHEMBL_SDF, OUTPUT_PKL, num_cpus)

    # Aşama B: Decoy adaylarını seç
    if chembl_full_data_list:
        run_selection(BINDER_PROPS_CSV, chembl_full_data_list, OUTPUT_CANDIDATES_CSV, num_cpus)

    elapsed = time.time() - start_time
    print(f"\n--- DECOY ADAYI HAZIRLIĞI TAMAMLANDI ---")
    print(f"Toplam süre: {elapsed:.2f} saniye")
