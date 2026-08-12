from operator import index
import torch
import numpy as np
import pickle,argparse,os,math,rdkit
import rdkit.Chem as Chem
from torch.utils.data import Dataset,DataLoader
import sys
import os
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from chemutils import ProteinPocket,mol_batch_to_graph,tensorize_protein_pocket

# PLANET_model.py dosyasının bulunduğu dizinin yolu
PLANET_MODEL_DIR = os.path.join(PLANET_ROOT, "Model")

# Bu dizini Python'ın arama yoluna ekle
if PLANET_MODEL_DIR not in sys.path:
    sys.path.append(PLANET_MODEL_DIR)

# Şimdi import et:
from PLANET_model import PLANET
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



class VirtualScreening_Dataset(Dataset):
    def __init__(self,smi_file,batch_size=32):
        self.batch_size = batch_size
        with open(smi_file,'r') as f:
            smi_content = [line.strip().split()[0] for line in f]
        self.smi_content = [smi_content[i:i+batch_size] for i in range(0,len(smi_content),batch_size)]

    def __len__(self):
        return len(self.smi_content)

    def __getitem__(self,idx):
        return self.tensorize(idx)

    def tensorize(self,idx):
        mol_smi_batch = self.smi_content[idx]
        mol_batch = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in mol_smi_batch if Chem.MolFromSmiles(smi) is not None]
        mol_feature_batch = mol_batch_to_graph(mol_batch,auto_detect=False)
        mol_smiles = [smi for smi in mol_smi_batch if Chem.MolFromSmiles(smi) is not None]
        return (mol_feature_batch,mol_smiles)

def predict(model:PLANET,protein_pdb,ligand_sdf,dataloader):
    try:
        pocket = ProteinPocket(protein_pdb=protein_pdb,ligand_sdf=ligand_sdf)
    except ValueError as e:
        # Hatalı/bozuk dosyayı ve boş pocket'i atla
        print(f"UYARI: {ligand_sdf} pocket hatası nedeniyle atlandı: {e}", file=sys.stderr)
        return np.array([]), [] # Boş sonuçlar döndür
    
    fresidues = model.cal_res_features_helper(pocket.res_features,pocket.alpha_coordinates)
    predicted_affinities,smiles = [],[]
    for (mol_feature_batch,mol_smiles) in dataloader:
        fresidues_batch,res_scope = model.cal_res_features(fresidues,len(mol_smiles))
        predicted_affinity = model.screening(fresidues_batch,res_scope,mol_feature_batch)
        predicted_affinities.append(np.array(predicted_affinity.squeeze().detach().cpu().reshape([-1])))
        smiles.extend(mol_smiles)
    predicted_affinities = np.concatenate(predicted_affinities,axis=0)
    return predicted_affinities,smiles
# Sadece workflow fonksiyonunu değiştirmeniz yeterlidir:

def workflow(model,folder_path,out_folder):
    # DÜZELTME 1: actives_prep.smi yerine mevcut 'actives_final.ism' dosyasını kullan
    actives_smi = os.path.join(folder_path,'actives_final.ism') 
    
    # DÜZELTME 2: decoys_prep.smi yerine mevcut 'decoys_final.ism' dosyasını kullan
    decoys_smi = os.path.join(folder_path,'decoys_final.ism')
    
    # DÜZELTME: crystal_ligand.mol2 yerine YENİ OLUŞTURULAN .sdf dosyasını kullan
    crystal_ligand = os.path.join(folder_path,'crystal_ligand.sdf') 
    
    # DÜZELTME: prepared_receptor.pdb yerine mevcut receptor.pdb dosyasını kullan
    pdb_path = os.path.join(folder_path,'receptor.pdb')

    # Not: Bu aşamada, ProteinPocket sınıfınızın .mol2 dosyasını okuyabildiği varsayılır. 
    # (RDKit MolSupplier .mol2 okuyabilir, bu yüzden bu çalışmalıdır.)

    active_dataset = VirtualScreening_Dataset(actives_smi)
    active_loader = DataLoader(active_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])
    
    decoy_dataset = VirtualScreening_Dataset(decoys_smi)
    decoy_loader = DataLoader(decoy_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])
    
    # Tahmin yap
    predicted_active_affinities, active_smiles = predict(model, pdb_path, crystal_ligand, active_loader)
    predicted_decoy_affinities, decoy_smiles = predict(model, pdb_path, crystal_ligand, decoy_loader)

    # 🌟 KRİTİK DÜZELTME: BOŞ SONUÇ KONTROLÜ 🌟
    # Eğer aktifler veya decoylar boşsa, bu hedef atlanmıştır (Veri bozuktur).
    if predicted_active_affinities.size == 0 or predicted_decoy_affinities.size == 0:
        print(f"BİLGİ: {folder_path} hedefi boş sonuçlar nedeniyle AUC hesaplaması için atlandı.")
        # EF_sort'a boş diziler göndermek yerine, doğrudan boş sonuçlar döndür
        return np.array([]), np.array([]) # Veya None döndürerek main döngüsünde kontrol edin.
    
    active_path = os.path.join(out_folder,'active.csv')
    decoy_path = os.path.join(out_folder,'decoy.csv')
    write_csv(predicted_active_affinities,active_smiles,active_path)
    write_csv(predicted_decoy_affinities,decoy_smiles,decoy_path)
    
    return predicted_active_affinities,predicted_decoy_affinities

def write_csv(affinities,smiles,out_path):
    csv_frame = pd.DataFrame([
        {
            'Affinity':affinity,'SMILES':smi,
        }
        for affinity,smi in zip(affinities,smiles)
    ])
    csv_frame.to_csv(out_path)

def EF_sort(active_pK,decoy_pK):
    all_pk = np.concatenate([active_pK,decoy_pK])
    cat_array = np.concatenate([np.ones_like(active_pK),np.zeros_like(decoy_pK)])
    auc_score = roc_auc_score(cat_array,all_pk)
    sorted_index = np.argsort(all_pk*(-1))
    sorted_array = np.array([cat_array[index] for index in sorted_index])
    return sorted_array,auc_score

def EF_calculate(sorted_array,factor):
    all_count = len(sorted_array)
    sample_count = math.ceil(all_count*factor)
    hits_total = sorted_array.sum()
    hits_count = sorted_array[:sample_count].sum()
    EF = (hits_count/sample_count) / (hits_total/all_count)
    return EF

def _percision_recall_area(active_pK,decoy_pK):
    y_pred = np.concatenate([active_pK,decoy_pK])
    y_true = np.concatenate([np.ones_like(active_pK),np.zeros_like(decoy_pK)])
    precision, recall, thresholds = precision_recall_curve(y_true,y_pred)
    area = auc(recall, precision)
    return area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--PLANET_file',required=True)
    parser.add_argument('-d','--DUDE',required=True)
    parser.add_argument('-o','--out_dir',required=True)
    
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
    args = parser.parse_args()
    
    # Argümanlardan gerekli boyutları çıkar (Bu kısım zaten kodunuzda vardı)
    feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters = \
        args.feature_dims, args.nheads, args.key_dims, args.value_dims, \
        args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters

    # Cihazı kesinlikle CPU olarak ayarla
    device = torch.device('cpu') 
    
    # 1. Modeli Başlat (device='cpu' ile)
    PLANET = PLANET(feature_dims, nheads, key_dims, value_dims, 
                      pro_update_inters, lig_update_iters, pro_lig_update_iters, 
                      'cpu').to(device) 
    
    # 2. Ağırlıkları Yükle (GPU'da eğitilmiş olsa bile CPU'ya eşle)
    PLANET.load_state_dict(torch.load(args.PLANET_file, map_location=device))
    
    # Sonuçların kaydedileceği dizini oluşturma
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    target_names,active_means,active_medians,decoy_means,decoy_medians,EF_05,EF_1,EF_2,EF_5,auc_scores,auc_prs = [],[],[],[],[],[],[],[],[],[],[]
    PLANET.eval()
    with torch.no_grad():
        for folder in os.listdir(args.DUDE):
            work_folder = os.path.join(args.DUDE,folder)
            
            # Yalnızca dizinleri işle
            if not os.path.isdir(work_folder):
                continue
                
            out_folder = os.path.join(out_dir,folder)
            
            os.makedirs(out_folder,exist_ok=True)
            
            # workflow çağrısı
            predicted_active_affinities,predicted_decoy_affinities = workflow(model=PLANET,folder_path=work_folder,out_folder=out_folder)
            
            # 🌟 KRİTİK KONTROL: Boş dizi kontrolü 🌟
            # predicted_active_affinities ve predicted_decoy_affinities,
            # workflow'daki atlama mekanizması nedeniyle boş diziler (size=0) olabilir.
            if predicted_active_affinities.size == 0 or predicted_decoy_affinities.size == 0:
                print(f"BİLGİ: {folder} hedefi atlandı. AUC hesaplaması için yeterli veri yok.")
                continue # Döngünün bir sonraki iterasyonuna geç
            
            # Eğer diziler doluysa, güvenle metrikleri hesapla:
            sorted_array,auc_score = EF_sort(predicted_active_affinities,predicted_decoy_affinities)
            auc_pr = _percision_recall_area(predicted_active_affinities,predicted_decoy_affinities)
            
            # Sonuçları Listelere Ekleme
            target_names.append(str(folder).upper())
            active_means.append(predicted_active_affinities.mean())
            active_medians.append(np.median(predicted_active_affinities))
            decoy_means.append(predicted_decoy_affinities.mean())
            decoy_medians.append(np.median(predicted_decoy_affinities))
            EF_05.append(EF_calculate(sorted_array,0.005))
            EF_1.append(EF_calculate(sorted_array,0.01))
            EF_2.append(EF_calculate(sorted_array,0.02))
            EF_5.append(EF_calculate(sorted_array,0.05))
            auc_scores.append(auc_score)
            auc_prs.append(auc_pr)
            torch.cuda.empty_cache() # CPU kullanılıyor olsa bile bu kalabilir.

        # Tüm hedefler işlendikten sonra özet tabloyu oluştur.
        # Bu kısımda zip döngüsünde bir değişken eksikliği vardı (auc_prs), onu düzelttik.
        csv_frame = pd.DataFrame([
            {
                'Target Name':target_name,'Active Mean':active_mean,'Active Median':active_median,
                'Decoy Mean':decoy_mean,'Decoy Median':decoy_median,'Delta Mean':active_mean-decoy_mean,'Delta Median':active_median-decoy_median,
                'EF_0.5%':x,'EF_1%':y,'EF_2%':z,'EF_5%':w,'ROC_AUC Score':auc_score,'ROC_PR Score':auc_pr
            }
            for target_name,active_mean,active_median,decoy_mean,decoy_median,x,y,z,w,auc_score,auc_pr in zip(target_names,active_means,active_medians,decoy_means,decoy_medians,EF_05,EF_1,EF_2,EF_5,auc_scores,auc_prs)
        ])
        csv_frame.to_csv(os.path.join(out_dir,'summary.csv'))







