# inference_new_data.py
import sys
import os

# --- KRİTİK DÜZELTME: Dosya Yollarını Tanıtma ---
# Python normalde sadece çalıştığı klasöre bakar. 
# Aşağıdaki kodlarla ona "Şu klasörlere de bak, orada aradığım dosyalar var" diyoruz.

# 1. PLANET_model.py ve layers.py'nin olduğu klasör:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. chemutils.py'nin olduğu klasör:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------

import torch
import numpy as np
import argparse
import pandas as pd
import rdkit.Chem as Chem
from torch.utils.data import Dataset, DataLoader

# Artık importlar hata vermez, çünkü yukarıda adresleri verdik
try:
    from PLANET_model import PLANET
    from chemutils import ProteinPocket, mol_batch_to_graph
    print("✅ Gerekli PLANET modülleri başarıyla yüklendi.")
except ImportError as e:
    print(f"❌ HATA: Modüller hala bulunamıyor.\nDetay: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEYİCİ SINIF (DUDE.py'den uyarlandı)
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    """
    SMILES formatındaki molekülleri okur ve PLANET modelinin anlayacağı
    Graph formatına çevirir.
    """
    def __init__(self, input_file, batch_size=32):
        self.batch_size = batch_size
        self.mol_data = [] # (SMILES, ID) tutacak

        # Dosya uzantısına göre okuma yapalım
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
            # CSV içinde 'smiles' veya 'SMILES' sütunu arıyoruz
            smi_col = [c for c in df.columns if 'smiles' in c.lower()]
            if not smi_col:
                print("HATA: CSV dosyasında 'smiles' isimli bir sütun bulunamadı.")
                sys.exit(1)
            
            # SMILES ve (varsa) ID'leri listeye at
            for index, row in df.iterrows():
                smi = row[smi_col[0]]
                # ID sütunu varsa al, yoksa satır numarasını ID yap
                mol_id = row['id'] if 'id' in row else f"mol_{index}"
                self.mol_data.append((smi, mol_id))

        else: # .smi veya .ism veya .txt dosyası varsayımı
            with open(input_file, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts: continue
                    smi = parts[0]
                    mol_id = parts[1] if len(parts) > 1 else f"mol_{idx}"
                    self.mol_data.append((smi, mol_id))
        
        # Veriyi batch'lere (paketlere) bölüyoruz
        self.batches = [self.mol_data[i:i + batch_size] for i in range(0, len(self.mol_data), batch_size)]
        print(f"📂 Toplam {len(self.mol_data)} molekül, {len(self.batches)} batch halinde yüklendi.")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_data = self.batches[idx]
        valid_mols = []
        valid_smiles = []
        valid_ids = []

        for (smi, mid) in batch_data:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                m = Chem.AddHs(m) # Hidrojenleri ekle (Model için kritik)
                valid_mols.append(m)
                valid_smiles.append(smi)
                valid_ids.append(mid)
        
        if len(valid_mols) == 0:
            return None # Boş batch dönerse collate_fn bunu yönetmeli

        # chemutils.py içindeki fonksiyon ile graf'a çevir
        # auto_detect=False çünkü SMILES'tan okuduk, koordinat yok (2D)
        mol_feature_batch = mol_batch_to_graph(valid_mols, auto_detect=False)
        
        return (mol_feature_batch, valid_smiles, valid_ids)

# -----------------------------------------------------------------------------
# 3. TAHMİN FONKSİYONU
# -----------------------------------------------------------------------------
def run_inference(model, protein_pdb, ref_ligand, dataloader, device):
    """
    Tek bir protein cebi için tüm ligand listesini tarar.
    """
    print(f"🧬 Protein İşleniyor: {os.path.basename(protein_pdb)}")
    print(f"📍 Referans Ligand (Cep Merkezi için): {os.path.basename(ref_ligand)}")

    try:
        # Protein cebini oluştur (chemutils.py kullanıyor)
        # Bu işlem cepteki amino asit özelliklerini çıkarır.
        pocket = ProteinPocket(protein_pdb=protein_pdb, ligand_sdf=ref_ligand)
    except Exception as e:
        print(f"❌ KRİTİK HATA: Protein cebi oluşturulamadı. Referans ligand dosyası bozuk veya protein ile uyumsuz olabilir.\nDetay: {e}")
        return []

    # Protein özelliklerini modele hazırlama
    # Modelin 'cal_res_features_helper' fonksiyonunu kullanıyoruz (PLANET_model.py)
    fresidues = model.cal_res_features_helper(pocket.res_features.to(device), pocket.alpha_coordinates.to(device))
    
    results = []
    
    model.eval() # Modeli değerlendirme moduna al
    with torch.no_grad(): # Gradyan hesaplamayı kapat (Hız ve hafıza için)
        for batch_idx, data in enumerate(dataloader):
            if data is None: continue
            
            mol_feature_batch, smiles_list, ids_list = data
            
            # Ligand graf verilerini cihaza (CPU/GPU) taşıma işi genelde model içinde veya helper'da yapılır
            # Ancak burada manuel bir taşıma gerekebilir, PLANET_model.py yapısına güveniyoruz.
            # PLANET screening fonksiyonu bunu hallediyor.

            # Proteini batch boyutu kadar çoğalt (Broadcasting)
            fresidues_batch, res_scope = model.cal_res_features(fresidues, len(smiles_list))
            
            # TAHMİN ANI
            # screening fonksiyonu çıktı olarak affinite skorlarını verir
            predicted_affinity = model.screening(fresidues_batch, res_scope, mol_feature_batch)
            
            # Sonuçları listeye ekle
            scores = predicted_affinity.squeeze().detach().cpu().numpy().reshape(-1)
            
            for i in range(len(scores)):
                results.append({
                    'ID': ids_list[i],
                    'SMILES': smiles_list[i],
                    'Predicted_Affinity': float(scores[i])
                })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   ... {batch_idx + 1} batch tamamlandı.")

    return results

# -----------------------------------------------------------------------------
# 4. MAIN BLOĞU
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLANET Modeli ile Yeni Veri Seti Test Aracı")
    
    # Gerekli Dosyalar
    parser.add_argument('--protein', required=True, help="Hedef proteinin PDB dosyası (örn: receptor.pdb)")
    parser.add_argument('--ref_ligand', required=True, help="Cebin yerini belirleyen referans ligand (SDF veya MOL2 formatında).")
    parser.add_argument('--ligands', required=True, help="Test edilecek ligandların listesi (.csv veya .smi). CSV ise 'smiles' sütunu olmalı.")
    parser.add_argument('--model', required=True, help="Eğitilmiş model dosyası (.iter veya .pth)")
    parser.add_argument('--output', default="results.csv", help="Sonuçların kaydedileceği CSV dosyası.")
    
    # Model Hiperparametreleri (Eğitimdeki ile AYNI olmalı)
    parser.add_argument('--feature_dims', type=int, default=300) # Varsayılan: 300
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--key_dims', type=int, default=300) # Eğitimdeki default değerler
    parser.add_argument('--value_dims', type=int, default=300)
    
    args = parser.parse_args()

    # Cihaz Ayarı (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Çalışma Ortamı: {device}")

    # 1. Modeli Yükle
    print("🔄 Model Yükleniyor...")
    try:
        # Model yapısını başlat (DUDE.py parametreleriyle aynı)
        planet_model = PLANET(
            feature_dims=args.feature_dims,
            nheads=args.nheads,
            key_dims=args.key_dims,
            value_dims=args.value_dims,
            pro_update_inters=3,     # Varsayılan
            lig_update_iters=10,     # Varsayılan
            pro_lig_update_iters=1,  # Varsayılan
            device=device
        ).to(device)

        # Ağırlıkları yükle
        # map_location, GPU'da eğitilip CPU'da çalıştırılıyorsa hatayı önler
        planet_model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"✅ Model ağırlıkları yüklendi: {args.model}")
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        print("İpucu: Hiperparametreler (feature_dims vs.) eğitimdekiyle aynı mı?")
        sys.exit(1)

    # 2. Veri Setini Hazırla
    print("🔄 Ligandlar Hazırlanıyor...")
    test_dataset = InferenceDataset(args.ligands, batch_size=32)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, # Collate fonksiyonu dataset içinde halledildiği için batch_size burada 1 olmalı
        shuffle=False, 
        num_workers=0, # Hata alırsan 0 yap, hız için 4 yapabilirsin
        collate_fn=lambda x: x[0]
    )

    # 3. Testi Başlat
    all_results = run_inference(planet_model, args.protein, args.ref_ligand, test_loader, device)

    # 4. Sonuçları Kaydet
    if all_results:
        df_res = pd.DataFrame(all_results)
        # Skorları yüksekten düşüğe sırala (Genelde yüksek skor = yüksek afinite)
        # NOT: Modelin çıktısı pK değeri ise büyük olan iyidir.
        df_res = df_res.sort_values(by='Predicted_Affinity', ascending=False)
        
        df_res.to_csv(args.output, index=False)
        print(f"✅ İşlem Tamamlandı! Sonuçlar kaydedildi: {args.output}")
        print(f"   En iyi skor: {df_res.iloc[0]['Predicted_Affinity']:.4f}")
    else:
        print("⚠️ Hiçbir sonuç üretilemedi. Girdi dosyalarını kontrol edin.")
        


'''
python inference_new_data.py \
  --protein ALDH1.pdb \
  --ref_ligand ligand_referans.sdf \
  --ligands test_molecules.csv \
  --model PLANET.iter-50000 \
  --output ALDH1_sonuclar.csv '''
