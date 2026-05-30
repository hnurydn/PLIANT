import os
import json
import warnings
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

# BioPython uyarılarını sustur
warnings.filterwarnings("ignore")

# ==========================================
#               AYARLAR
# ==========================================

# JSON Dosya Yolları (Senin verdiğin yollar)
LIT_PCBA_JSON = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_arf.json"
DUDE_JSON = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/dude/dude_paths_prepared_arf.json"

# Çıktı Dosyası
OUTPUT_TEST_FASTA = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/test_sequences_for_esm2.fasta"

# ==========================================
#           YARDIMCI FONKSİYONLAR
# ==========================================

def get_sequence_from_pdb(pdb_path):
    """PDB dosyasından amino asit dizisini çıkarır (CA atomlarını takip eder)."""
    try:
        # Hızlı okuma için sadece CA atomlarını manuel parse edelim (BioPython bazen yavaştır)
        # Ama senin yapın BioPython üzerine kurulu, onu koruyalım.
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_path)
        for model in structure:
            for chain in model:
                seq = []
                for residue in chain:
                    # Heteroatomları atla (H_ ile başlayanlar)
                    if residue.id[0] == ' ':
                        try:
                            # 3 harfli kodu 1 harfli koda çevir
                            res_name = residue.get_resname()
                            # Özel durumlar
                            if res_name == "MSE": res_name = "MET"
                            if res_name == "HID": res_name = "HIS"
                            if res_name == "HIE": res_name = "HIS"
                            if res_name == "HIP": res_name = "HIS"
                            
                            seq.append(seq1(res_name))
                        except: pass
                if len(seq) > 0: return "".join(seq)
    except Exception as e:
        # print(f"Hata PDB: {pdb_path} -> {e}")
        return None
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
                                if resi_name == "MSE": resi_name = "MET"
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
#           ANA İŞLEM FONKSİYONU
# ==========================================

def process_test_sets():
    """Test Setleri (LIT-PCBA ve DUD-E) için ortak FASTA oluşturur."""
    print(f"\n🔹 Test Setleri FASTA Hazırlanıyor...")
    print(f"Hedef Dosya: {OUTPUT_TEST_FASTA}")
    
    unique_headers = set()
    count_lit = 0
    count_dude = 0
    
    with open(OUTPUT_TEST_FASTA, "w") as out_f:
        
        # --- 1. LIT-PCBA ---
        if os.path.exists(LIT_PCBA_JSON):
            with open(LIT_PCBA_JSON, 'r') as f:
                data = json.load(f)
            
            print(f"🚀 LIT-PCBA ({len(data)} targets) işleniyor...")
            for target, pdb_list in tqdm(data.items(), desc="LIT-PCBA"):
                if not pdb_list: continue
                # Genelde LIT-PCBA'da her target için tek protein yapısı kullanılır
                # İlk geçerli olanı almamız yeterli olabilir ama hepsini tarayalım.
                
                # Ancak burada dikkat: LIT-PCBA yapısında `pdb_list` bir liste mi yoksa dict mi?
                # Genelde listedir.
                for entry in pdb_list:
                    name = entry.get("name", "unknown")
                    pdb_path = entry.get("protein_pdb")
                    mol2_path = entry.get("protein_mol2")
                    
                    # Sequence Çıkar
                    seq = None
                    if pdb_path and os.path.exists(pdb_path):
                        seq = get_sequence_from_pdb(pdb_path)
                    
                    if not seq and mol2_path and os.path.exists(mol2_path):
                        seq = get_sequence_from_mol2(mol2_path)
                        
                    if seq:
                        # HEADER ÖNEMLİ: Daha sonra mapping yaparken bu ID'yi kullanacağız.
                        # Benzersiz olması için Target adını kullanıyoruz.
                        # LIT-PCBA'da genelde target başına 1 protein olur.
                        header = f"LITPCBA_{target}" 
                        
                        if header not in unique_headers:
                            out_f.write(f">{header}\n{seq}\n")
                            unique_headers.add(header)
                            count_lit += 1
                        
                        # Sequence bulduysak bu target için diğer entrylere bakmaya gerek yok (Genelde aynıdır)
                        break 
        else:
            print(f"❌ UYARI: LIT-PCBA JSON dosyası bulunamadı: {LIT_PCBA_JSON}")

        # --- 2. DUD-E ---
        if os.path.exists(DUDE_JSON):
            with open(DUDE_JSON, 'r') as f:
                data = json.load(f)
            
            print(f"🚀 DUD-E ({len(data)} targets) işleniyor...")
            for target, pdb_list in tqdm(data.items(), desc="DUD-E"):
                # DUD-E yapısı: Key=TargetName (örn: aa2ar), Value=[{rec}]
                if not pdb_list: continue
                entry = pdb_list[0] # İlk kaydı al
                
                pdb_path = entry.get("protein_pdb")
                # Mol2 path bazen json'da olmayabilir, türetelim
                mol2_path = entry.get("protein_mol2")
                if not mol2_path and pdb_path:
                    mol2_path = pdb_path.replace(".pdb", ".mol2")

                seq = None
                if pdb_path and os.path.exists(pdb_path):
                    seq = get_sequence_from_pdb(pdb_path)
                
                if not seq and mol2_path and os.path.exists(mol2_path):
                    seq = get_sequence_from_mol2(mol2_path)

                if seq:
                    header = f"DUDE_{target}" # Örn: DUDE_aa2ar
                    if header not in unique_headers:
                        out_f.write(f">{header}\n{seq}\n")
                        unique_headers.add(header)
                        count_dude += 1
        else:
             print(f"❌ UYARI: DUD-E JSON dosyası bulunamadı: {DUDE_JSON}")
             
    print(f"\n✅ İŞLEM TAMAMLANDI.")
    print(f"   - LIT-PCBA Eklenen: {count_lit}")
    print(f"   - DUD-E Eklenen   : {count_dude}")
    print(f"   - Toplam Sequence : {count_lit + count_dude}")
    print(f"📄 Çıktı Dosyası: {OUTPUT_TEST_FASTA}")

if __name__ == "__main__":
    process_test_sets()