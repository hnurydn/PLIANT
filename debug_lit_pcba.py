import json
import os
import sys
import traceback
from rdkit import Chem
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



# 1. Chemutils Patchli Versiyonu Yukle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from chemutils_dude import ComplexPocket
    print("✅ chemutils_dude (PATCHED) yuklendi.")
except ImportError:
    print("❌ chemutils_dude BULUNAMADI! Patch scripti calistirilmamis.")
    sys.exit(1)

# Ayarlar
TEST_TARGET = "ADRB2"
JSON_FILE = os.path.join(PLANET_ROOT, "datasets/pdbbind2020/LIT-PCBA/lit_pcba_paths_prepared_pdb.json")

def main():
    print(f"\n🔍 DEBUG BASLIYOR: {TEST_TARGET} hedefini test ediyoruz...")
    
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON Dosyasi Yok: {JSON_FILE}")
        return

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    
    if TEST_TARGET not in data:
        print(f"❌ JSON icinde {TEST_TARGET} bulunamadi!")
        return

    record = data[TEST_TARGET][0]
    protein_path = record['protein_file']
    actives_path = record['actives_file']
    
    print(f"📂 Protein Yolu: {protein_path}")
    print(f"📂 Actives Yolu: {actives_path}")

    # 1. Dosyalarin Varligini Kontrol Et
    if not os.path.exists(protein_path):
        print("❌ HATA: Protein dosyasi fiziksel olarak yok!")
    else:
        print("✅ Protein dosyasi mevcut.")

    if not os.path.exists(actives_path):
        print("❌ HATA: Actives SDF dosyasi fiziksel olarak yok! (SDF uretme adimi basarisiz olmus olabilir)")
        return
    else:
        print("✅ Actives SDF dosyasi mevcut.")

    # 2. SDF Okuma Testi
    print("\n🧪 Test 1: SDF Dosyasi Okunuyor...")
    mols = []
    try:
        suppl = Chem.SDMolSupplier(actives_path, sanitize=False)
        for i, m in enumerate(suppl):
            if i >= 1: break # Sadece ilk molekule bak
            if m:
                try:
                    Chem.SanitizeMol(m)
                    mols.append(m)
                    print(f"✅ Molekul {i} okundu: {m.GetNumAtoms()} atom")
                except:
                    print(f"⚠️ Molekul {i} Sanitize edilemedi!")
            else:
                print(f"❌ Molekul {i} RDKit tarafindan okunamadi (None dondu).")
    except Exception as e:
        print(f"❌ SDF Okuma Hatasi: {e}")
        return

    if not mols:
        print("❌ Hicbir molekul basariyla okunamadi! SDF dosyasi bozuk veya bos olabilir.")
        return
    
    # 3. ComplexPocket Olusturma Testi
    print("\n🧪 Test 2: ComplexPocket olusturuluyor...")
    mol = mols[0]
    try:
        # Patchli chemutils dosya yolu yerine Mol objesi kabul etmeli
        pocket = ComplexPocket(protein_path, mol, pK=5.0)
        print("🎉 BASARILI! ComplexPocket olustu.")
        print(f"Cep bilgisi: {pocket}")
    except Exception as e:
        print("❌ ComplexPocket HATASI:")
        print("-" * 30)
        traceback.print_exc()
        print("-" * 30)

if __name__ == "__main__":
    main()
