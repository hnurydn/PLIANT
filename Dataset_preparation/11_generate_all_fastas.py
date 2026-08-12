import os
import json
import warnings
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm
ANZU_DATA_ROOT = os.environ.get("ANZU_DATA_ROOT", ".")  # local data checkout dir



# BioPython uyarılarını sustur
warnings.filterwarnings("ignore")

# ==========================================
#               AYARLAR
# ==========================================

# 1. TRAIN SET (PDBbind) YOLLARI
PDBBIND_JSON = os.path.join(ANZU_DATA_ROOT, "PDBbind_structure_map_pkd_prepared.json")
OUTPUT_PDBBIND_FASTA = os.path.join(ANZU_DATA_ROOT, "pdbbind_sequences.fasta")

# 2. TEST SETLERİ (LIT-PCBA ve DUD-E) YOLLARI
LIT_PCBA_JSON = os.path.join(ANZU_DATA_ROOT, "LIT-PCBA/lit_pcba_paths_all_pdbs.json")
DUDE_JSON = os.path.join(ANZU_DATA_ROOT, "all/dude_paths_all_pdbs.json")
OUTPUT_TEST_FASTA = os.path.join(ANZU_DATA_ROOT, "test_sequences_all_variants.fasta")


# ==========================================
#           YARDIMCI FONKSİYONLAR
# ==========================================

def get_sequence_from_pdb(pdb_path):
    """PDB dosyasından amino asit dizisini çıkarır."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_path)
        for model in structure:
            for chain in model:
                seq = []
                for residue in chain:
                    # Heteroatomları atla (H_ ile başlayanlar)
                    if residue.id[0] == ' ':
                        try:
                            seq.append(seq1(residue.get_resname()))
                        except: pass
                if len(seq) > 0: return "".join(seq)
    except: return None
    return None

def get_sequence_from_mol2(mol2_path):
    """MOL2 dosyasını text olarak okuyup amino asit dizisini çıkarır."""
    try:
        residues = []
        last_resi_id = -1
        
        with open(mol2_path, 'r') as f:
            lines = f.readlines()
            
        is_atom_section = False
        for line in lines:
            if line.strip() == "@<TRIPOS>ATOM":
                is_atom_section = True
                continue
            elif line.strip().startswith("@<TRIPOS>"):
                is_atom_section = False
            
            if is_atom_section:
                parts = line.split()
                if len(parts) >= 8: # En az 8 sütun olmalı
                    try:
                        resi_id = int(parts[6])
                        resi_name_full = parts[7] # Örn: ALA1
                        resi_name = resi_name_full[:3] # ALA
                        
                        if resi_id != last_resi_id:
                            try:
                                one_letter = seq1(resi_name)
                                if len(one_letter) == 1: 
                                    residues.append(one_letter)
                            except: pass
                            last_resi_id = resi_id
                    except: continue

        if len(residues) > 0: return "".join(residues)
    except: return None
    return None


# ==========================================
#           ANA İŞLEM FONKSİYONLARI
# ==========================================

def process_pdbbind():
    """PDBbind (Eğitim Seti) için FASTA oluşturur."""
    print(f"\n🔹 PDBbind FASTA Hazırlanıyor...")
    
    if not os.path.exists(PDBBIND_JSON):
        print(f"HATA: {PDBBIND_JSON} bulunamadı!")
        return

    with open(PDBBIND_JSON, 'r') as f:
        data = json.load(f)

    count_success = 0
    
    with open(OUTPUT_PDBBIND_FASTA, "w") as out_f:
        for pdb_id, info in tqdm(data.items(), desc="PDBbind İşleniyor"):
            original_path = info.get('protein_pdb', '')
            if not original_path: continue
            
            # _prepared.pdb yerine _protein.pdb kullanımı
            real_path = original_path.replace("_protein_prepared.pdb", "_protein.pdb")
            
            if os.path.exists(real_path):
                seq = get_sequence_from_pdb(real_path)
                if seq:
                    out_f.write(f">{pdb_id}\n{seq}\n")
                    count_success += 1
    
    print(f"✅ PDBbind Tamamlandı: {count_success} dizi kaydedildi.")
    print(f"📄 Dosya: {OUTPUT_PDBBIND_FASTA}")


def process_test_sets():
    """Test Setleri (LIT-PCBA ve DUD-E) için ortak FASTA oluşturur."""
    print(f"\n🔹 Test Setleri FASTA Hazırlanıyor...")
    
    unique_headers = set()
    count_lit = 0
    count_dude = 0
    
    with open(OUTPUT_TEST_FASTA, "w") as out_f:
        
        # --- 1. LIT-PCBA ---
        if os.path.exists(LIT_PCBA_JSON):
            with open(LIT_PCBA_JSON, 'r') as f:
                data = json.load(f)
            
            for target, pdb_list in tqdm(data.items(), desc="LIT-PCBA İşleniyor"):
                for entry in pdb_list:
                    name = entry.get("name")
                    pdb_path = entry.get("protein_pdb")
                    mol2_path = entry.get("protein_mol2")
                    
                    # Öncelik PDB, yoksa MOL2
                    seq = None
                    if pdb_path and os.path.exists(pdb_path):
                        seq = get_sequence_from_pdb(pdb_path)
                    if not seq and mol2_path and os.path.exists(mol2_path):
                        seq = get_sequence_from_mol2(mol2_path)
                        
                    if seq:
                        header = f"LITPCBA_{target}_{name}"
                        if header not in unique_headers:
                            out_f.write(f">{header}\n{seq}\n")
                            unique_headers.add(header)
                            count_lit += 1
        else:
            print("UYARI: LIT-PCBA JSON dosyası bulunamadı.")

        # --- 2. DUD-E ---
        if os.path.exists(DUDE_JSON):
            with open(DUDE_JSON, 'r') as f:
                data = json.load(f)
            
            for target, pdb_list in tqdm(data.items(), desc="DUD-E İşleniyor"):
                for entry in pdb_list:
                    name = entry.get("name") # Target name (aa2ar vb.)
                    pdb_path = entry.get("protein_pdb")
                    mol2_path = entry.get("protein_mol2") if "protein_mol2" in entry else None
                    # Eğer JSON'da mol2 yoksa tahmin et
                    if not mol2_path and pdb_path:
                        mol2_path = pdb_path.replace(".pdb", ".mol2")

                    seq = None
                    if pdb_path and os.path.exists(pdb_path):
                        seq = get_sequence_from_pdb(pdb_path)
                    if not seq and mol2_path and os.path.exists(mol2_path):
                        seq = get_sequence_from_mol2(mol2_path)

                    if seq:
                        header = f"DUDE_{target}"
                        if header not in unique_headers:
                            out_f.write(f">{header}\n{seq}\n")
                            unique_headers.add(header)
                            count_dude += 1
        else:
             print("UYARI: DUD-E JSON dosyası bulunamadı.")
             
    print(f"✅ Test Setleri Tamamlandı.")
    print(f"   - LIT-PCBA Girişi: {count_lit}")
    print(f"   - DUD-E Girişi: {count_dude}")
    print(f"📄 Dosya: {OUTPUT_TEST_FASTA}")


# ==========================================
#               MAIN
# ==========================================

if __name__ == "__main__":
    process_pdbbind()
    process_test_sets()
    
    print("\n🎉 TÜM FASTA DOSYALARI HAZIR!")
    print("Sıradaki Adım: Terminalden şu komutu çalıştırarak sızıntı kontrolü yap:")
    print(f"cd-hit-2d -i {OUTPUT_TEST_FASTA} -i2 {OUTPUT_PDBBIND_FASTA} -o train_overlaps -c 0.90 -n 5")