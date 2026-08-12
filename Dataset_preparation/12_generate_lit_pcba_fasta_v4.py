import os, json, warnings, re
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



warnings.filterwarnings("ignore")
AA_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

# --- YOLLAR ---
LIT_PCBA_JSON = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_arf.json")
OUTPUT_FASTA = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA/lit_pcba_sequences_full.fasta")

def get_multi_chain_sequence_pdb(pdb_path):
    """Tüm zincirleri birleştirerek dizi çıkarır."""
    sequences = []
    try:
        with open(pdb_path, 'r') as f:
            current_res = None
            for line in f:
                if line.startswith("ATOM"):
                    res_num = line[22:26].strip()
                    chain_id = line[21].strip() or 'A'
                    res_type = line[17:20].strip()
                    res_id = (chain_id, res_num)
                    if res_id != current_res and res_type in AA_3_TO_1:
                        sequences.append(AA_3_TO_1[res_type])
                        current_res = res_id
        return "".join(sequences) if sequences else None
    except: return None

def get_multi_chain_sequence_mol2(mol2_path):
    """MOL2 dosyasından tüm diziyi çıkarır."""
    residues = []
    try:
        last_id = -1
        with open(mol2_path, 'r') as f:
            is_atom = False
            for line in f:
                if "@<TRIPOS>ATOM" in line: is_atom = True; continue
                if "@<TRIPOS>" in line and is_atom: is_atom = False
                if is_atom:
                    parts = line.split()
                    if len(parts) >= 8:
                        rid, rname = int(parts[6]), parts[7][:3]
                        if rid != last_id:
                            one = seq1(rname)
                            if len(one) == 1: residues.append(one)
                            last_id = rid
        return "".join(residues) if residues else None
    except: return None

def main():
    print(f"🚀 LIT-PCBA FASTA Hazırlanıyor (Multi-Chain & Name ID)...")
    with open(LIT_PCBA_JSON, 'r') as f: data = json.load(f)
    count = 0
    with open(OUTPUT_FASTA, "w") as out_f:
        for target, entries in tqdm(data.items()):
            for entry in entries:
                name = entry.get("name") # Örn: 3p0g
                path = entry.get("protein_file") or entry.get("protein_pdb")
                if not name or not path or not os.path.exists(path): continue
                seq = get_multi_chain_sequence_mol2(path) if path.endswith('.mol2') else get_multi_chain_sequence_pdb(path)
                if seq:
                    out_f.write(f">{name}\n{seq}\n")
                    count += 1
    print(f"✅ Bitti. {count} yapı kaydedildi: {OUTPUT_FASTA}")

if __name__ == "__main__": main()