import os
import sys
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# Dosya Yolları
ORIGINAL_FILE = os.path.join(PLANET_ROOT, "Dataset_preparation/chemutils.py")
NEW_FILE = os.path.join(PLANET_ROOT, "Dataset_preparation/chemutils_dude.py")

def main():
    print(f"🔧 Orijinal dosya okunuyor: {ORIGINAL_FILE}")
    
    if not os.path.exists(ORIGINAL_FILE):
        print("❌ HATA: Orijinal chemutils.py bulunamadı!")
        sys.exit(1)

    with open(ORIGINAL_FILE, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    fixed = False
    
    for line in lines:
        # Hatalı satırı yakalıyoruz (Dosya yolu bekleyen kısım)
        if "ligand = Chem.SDMolSupplier(ligand_sdf,removeHs=False)[0]" in line:
            indent = line.split("ligand")[0] # Girintiyi koru
            
            # YENİ MANTIK: String gelirse oku, Obje gelirse direkt kullan.
            new_lines.append(f"{indent}# --- PATCHED FOR DUD-E ---\n")
            new_lines.append(f"{indent}if isinstance(ligand_sdf, str):\n")
            new_lines.append(f"{indent}    ligand = Chem.SDMolSupplier(ligand_sdf,removeHs=False)[0]\n")
            new_lines.append(f"{indent}else:\n")
            new_lines.append(f"{indent}    ligand = ligand_sdf\n")
            new_lines.append(f"{indent}# -----------------------\n")
            fixed = True
        else:
            new_lines.append(line)
            
    if fixed:
        with open(NEW_FILE, 'w') as f:
            f.writelines(new_lines)
        print(f"🎉 Yeni dosya oluşturuldu: {NEW_FILE}")
    else:
        print("⚠️ Hatalı satır bulunamadı! Dosya zaten düzeltilmiş olabilir.")

if __name__ == "__main__":
    main()
