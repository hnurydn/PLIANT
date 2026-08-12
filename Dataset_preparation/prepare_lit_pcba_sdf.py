import json
import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- AYARLAR ---
# Makale standartlarina (Table S2) gore guncellenmis PDB JSON dosyasi
JSON_FILE = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_pdb.json")
BASE_DIR = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA")

# TEST MODU: Hizlica test etmek icin 1000 molekul. 
# Gercek egitim icin: LIMIT_MOLECULES = None
LIMIT_MOLECULES = 10000

def get_centroid(mol):
    try:
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        return np.mean(pos, axis=0)
    except:
        return None

def get_target_center(target_folder, protein_filename):
    """
    Protein dosyasina (orn: 4lde_protein.pdb) bakarak
    eslesen ligandi (orn: 4lde_ligand.mol2) bulur ve merkezini alir.
    """
    code = protein_filename.split('_')[0] 
    ligand_name = f"{code}_ligand.mol2"
    ligand_path = os.path.join(target_folder, ligand_name)
    
    # Eger tam isimle ligand yoksa klasordeki ilk _ligand.mol2 yi dene (Yedek)
    if not os.path.exists(ligand_path):
        for f in os.listdir(target_folder):
            if f.endswith("_ligand.mol2"):
                ligand_path = os.path.join(target_folder, f)
                break
    
    # Merkezi Bul
    if os.path.exists(ligand_path):
        # 1. RDKit ile dene
        mol = Chem.MolFromMol2File(ligand_path, sanitize=False)
        if mol: return get_centroid(mol)
        
        # 2. OpenBabel ile dene
        temp_pdb = ligand_path.replace(".mol2", "_temp_ref.pdb")
        os.system(f"obabel {ligand_path} -O {temp_pdb} 2>/dev/null")
        if os.path.exists(temp_pdb):
            mol = Chem.MolFromPDBFile(temp_pdb, sanitize=False)
            os.remove(temp_pdb)
            if mol: return get_centroid(mol)
            
    return None

def process_smiles_to_sdf(smiles_path, target_center, output_sdf_path):
    if os.path.exists(output_sdf_path) and os.path.getsize(output_sdf_path) > 1000:
        print(f"⏩ Zaten hazir: {os.path.basename(output_sdf_path)}")
        return

    print(f"⚙️  Isleniyor: {os.path.basename(smiles_path)} -> {os.path.basename(output_sdf_path)}")
    
    with open(smiles_path, 'r') as f:
        lines = f.readlines()

    writer = Chem.SDWriter(output_sdf_path)
    count = 0
    process_lines = lines[:LIMIT_MOLECULES] if LIMIT_MOLECULES else lines

    for line in tqdm(process_lines, desc="   Hizalaniyor"):
        parts = line.split()
        if not parts: continue
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol_{count}"

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            mol = Chem.AddHs(mol)
            mol.SetProp("_Name", name)

            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res != 0: continue 

            mol_center = get_centroid(mol)
            if mol_center is None: continue
            
            translation = target_center - mol_center
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, p + translation)

            writer.write(mol)
            count += 1
        except:
            continue

    writer.close()
    print(f"✅ {count} molekul kaydedildi.")

def main():
    if not os.path.exists(JSON_FILE):
        print("❌ JSON bulunamadi!")
        return

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    print(f"🚀 {len(data)} Hedef icin hizalama basliyor...")

    for target, records in data.items():
        if not records: continue
        
        target_folder = os.path.join(BASE_DIR, target)
        protein_full_path = records[0]['protein_file']
        protein_filename = os.path.basename(protein_full_path)
        
        # MERKEZI BUL (Makale standardina uygun proteinden)
        center = get_target_center(target_folder, protein_filename)
        
        if center is None:
            print(f"⚠️ UYARI: {target} icin merkez bulunamadi! Atlaniyor.")
            continue
            
        actives_smi = os.path.join(target_folder, "actives.smi")
        inactives_smi = os.path.join(target_folder, "inactives.smi")
        actives_sdf = os.path.join(target_folder, "actives.sdf")
        decoys_sdf = os.path.join(target_folder, "decoys.sdf")

        if os.path.exists(actives_smi):
            process_smiles_to_sdf(actives_smi, center, actives_sdf)
        if os.path.exists(inactives_smi):
            process_smiles_to_sdf(inactives_smi, center, decoys_sdf)

if __name__ == "__main__":
    main()
