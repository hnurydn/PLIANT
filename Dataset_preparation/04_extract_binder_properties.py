# 04_extract_binder_properties.py
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# --- Ayarlar ---
JSON_MAP_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/PDBbind_structure_map_pkd.json")
OUTPUT_CSV_PATH = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/binder_properties.csv")
# ----------------

def get_ligand_properties(mol):
    """Bir RDKit Mol objesi için makalede belirtilen özellikleri hesaplar."""
    if mol is None:
        return None
    try:
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        rings = Descriptors.RingCount(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        total_atoms = mol.GetNumAtoms()
        
        return {
            'MolWt': mol_weight,
            'LogP': logp,
            'NumHDonors': h_donors,
            'NumHAcceptors': h_acceptors,
            'NumChiralCenters': chiral_centers,
            'NumRings': rings,
            'NumHeavyAtoms': heavy_atoms,
            'NumTotalAtoms': total_atoms
        }
    except Exception as e:
        # Bazı moleküller (örn: organometalik) RDKit'te hata verebilir
        return None

def main():
    print(f"JSON haritası yükleniyor: {JSON_MAP_PATH}")
    if not os.path.exists(JSON_MAP_PATH):
        print(f"HATA: JSON haritası bulunamadı: {JSON_MAP_PATH}")
        return

    with open(JSON_MAP_PATH, 'r') as f:
        structure_map = json.load(f)

    print(f"Toplam {len(structure_map)} kompleks bulundu.")
    
    all_properties = []
    processed_count = 0
    failed_to_read_mol = 0
    pdbfile_not_run = 0
    
    # JSON haritasını TQDM ile (ilerleme çubuğu) işle
    for pdb_id, data in tqdm(structure_map.items(), desc="Ligandlar İşleniyor"):
        
        # --- YENİ MANTIK ---
        # 1. Gerekli dosya yolları JSON'da var mı?
        if 'protein_pdb' not in data or 'ligand_sdf' not in data:
            continue
            
        protein_path = data['protein_pdb']
        ligand_path = data['ligand_sdf']
        
        
        if os.path.exists(protein_path):
            # 3. Başarılıysa, ligandı oku
            if ligand_path and os.path.exists(ligand_path):
                try:
                    # Ligand dosyasını RDKit ile oku
                    if ligand_path.endswith('.sdf'):
                        suppl = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)
                        mol = next(suppl, None)
                    else:
                        # Diğer formatlar için (örn: mol2)
                        mol = Chem.MolFromMolFile(ligand_path, removeHs=False, sanitize=True)
                    
                    if mol:
                        properties = get_ligand_properties(mol)
                        if properties:
                            properties['pdb_id'] = pdb_id
                            properties['ligand_path'] = ligand_path
                            all_properties.append(properties)
                        else:
                            failed_to_read_mol += 1
                    else:
                        failed_to_read_mol += 1
                        
                except Exception as e:
                    # RDKit okuma sırasında kritik bir hata olursa
                    failed_to_read_mol += 1
            
            processed_count += 1
        else:
            # '..._protein_prepared.pdb' dosyası bulunamadı
            pdbfile_not_run += 1
            
    # --- SONUÇ KONTROLÜ ---
    
    if not all_properties:
        print("\nHiçbir ligand özelliği çıkarılamadı.")
        print(f"Toplam {processed_count} adet 'prepared' protein bulundu.")
        print(f"Toplam {pdbfile_not_run} adet 'prepared' protein bulunamadı.")
        print(f"Toplam {failed_to_read_mol} adet ligand RDKit tarafından okunamadı/işlenemedi.")
        return

    # Verileri bir DataFrame'e dönüştür
    df = pd.DataFrame(all_properties)
    
    # Sütun sırasını düzenle
    cols = ['pdb_id', 'ligand_path', 'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
            'NumChiralCenters', 'NumRings', 'NumHeavyAtoms', 'NumTotalAtoms']
    
    # Olası eksik sütunları doldur (eğer listede olmayan bir sütun varsa)
    df = df.reindex(columns=cols)

    # CSV olarak kaydet
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- İşlem Tamamlandı ---")
    print(f"Toplam {len(df)} adet başarılı ligandın özelliği çıkarıldı.")
    print(f"(Toplam {processed_count} adet 'prepared' protein bulundu.)")
    print(f"(Toplam {pdbfile_not_run} adet 'prepared' protein bulunamadı / atlandı.)")
    print(f"(Toplam {failed_to_read_mol} adet ligand RDKit tarafından okunamadı / atlandı.)")
    print(f"Sonuçlar kaydedildi: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()