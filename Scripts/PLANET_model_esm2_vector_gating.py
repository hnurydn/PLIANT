import math, os, sys, torch
import torch.nn as nn

# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers_esm2_vector_gating import ProteinEGNN, LigandGAT, ProLig
from nnutils import create_var, create_var_gpu

class PLANET(nn.Module):
    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dims = feature_dims
        
        # 1. Katmanlar
        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
        
        # ProLig: Artık geometrik hesaplamalar için koordinatları bekleyen yeni versiyonu çağırıyoruz
        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device)

        # 2. Uncertainty Weighting (Belirsizlik Ağırlıklandırma) Parametreleri
        # Train dosyasında loss'ları dinamik dengelemek için kullanılan öğrenilebilir sigma parametreleri.
        self.log_sigma_lig = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_pl = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_affinity = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)

        # 3. Kayıp Fonksiyonları
        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
        self.pro_lig_interaction_loss = nn.BCELoss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, res_batch, mol_batch):
        """
        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
        
        res_batch: Protein verisi (Özellikler, Harita, Scope, Koordinatlar)
        mol_batch: Ligand verisi (Atomlar, Bağlar, Graf, Scope, KOORDİNATLAR)
        """
        # --- A. Protein Verisini Hazırla ---
        # alpha_coordinates: Protein backbone koordinatları
        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
        fresidues = fresidues.to(self.device)
        alpha_coordinates = create_var(alpha_coordinates).to(self.device)

        # --- B. Ligand Verisini Hazırla ---
        # DİKKAT: Dataset'ten lig_coordinates'in (6. eleman) geldiğinden emin olmalıyız.
        # Eğer dataset 5 eleman döndürüyorsa 'PLANET_datautils_esm2.py' dosyasında collate_fn güncellenmelidir.
        if len(mol_batch) == 6:
            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
        else:
            # Hata yakalama: Eğer eski dataset kullanılıyorsa uyarı ver.
            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi. Lütfen datautils dosyasını kontrol edin.")

        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)

        # --- C. Alt Katmanlardan Geçiş ---
        
        # 1. Protein Embedding (EGNN - Koordinat kullanır)
        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
        
        # 2. Ligand Embedding (GAT - Graph tabanlı)
        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
         
        # 3. Protein-Ligand Etkileşimi (ProLig)
        # KRİTİK NOKTA: Hem protein (alpha) hem ligand koordinatlarını buraya paslıyoruz.
        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
        )
        
        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
    
    def compute_loss(self, predictions, targets, res_batch, mol_batch):
        """
        Ham kayıpları hesaplar. Ağırlıklandırma (Weighting) train dosyasında yapılır.
        """
        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions
        
        # Batch bilgilerini al
        (_, _, res_scope, _) = res_batch
        
        # mol_batch'ten lig_scope'u güvenli şekilde al (5. eleman)
        lig_scope = mol_batch[4] 

        # 6 elemanlı targets — group_ids'i al ama şimdilik kullanma
        if len(targets) == 6:
            mol_intearctions, pro_lig_interactions, pKs, pK_flags, complex_labels, group_ids = targets
        else:
            mol_intearctions, pro_lig_interactions, pKs, pK_flags, complex_labels = targets
        mol_intearctions     = create_var(mol_intearctions).to(self.device)
        pro_lig_interactions = create_var(pro_lig_interactions).to(self.device)
        pKs                  = create_var(pKs).to(self.device)
        pK_flags             = create_var(pK_flags).to(self.device)

        lig_interaction_count, pro_lig_interaction_count = 0, 0
        lig_interaction_loss, pro_lig_interaction_loss = [], []
        
        # Batch içindeki her bir kompleks için döngü
        for ((_, res_count), (_, atom_count), complex_label) in zip(res_scope, lig_scope, complex_labels):
            
            # 1. Ligand Interaction Loss
            # Payda atom_count^2 olmalı (etkileşim matrisi boyutu)
            lig_total_points = atom_count**2
            l_loss = self.ligand_interaction_loss(
                predicted_lig_interactions[lig_interaction_count : lig_interaction_count + lig_total_points],
                mol_intearctions[lig_interaction_count : lig_interaction_count + lig_total_points]
            ) / lig_total_points
            
            # 2. Protein-Ligand Interaction Loss
            # Payda atom_count * res_count olmalı
            pl_total_points = atom_count * res_count
            pl_loss = self.pro_lig_interaction_loss(
                predicted_interactions[pro_lig_interaction_count : pro_lig_interaction_count + pl_total_points],
                pro_lig_interactions[pro_lig_interaction_count : pro_lig_interaction_count + pl_total_points]
            ) / pl_total_points

            # Sadece geçerli kompleksler için loss ekle
            if complex_label == 1:
                lig_interaction_loss.append(l_loss)
            
            pro_lig_interaction_loss.append(pl_loss)
            
            # Sayaçları güncelle
            lig_interaction_count += lig_total_points
            pro_lig_interaction_count += pl_total_points
            
        # Ortalama Loss'lar
        lig_interaction_loss = torch.mean(torch.stack(lig_interaction_loss))
        pro_lig_interaction_loss = torch.mean(torch.stack(pro_lig_interaction_loss))
        
        # 3. Affinity Loss (Sadece pK değeri bilinenler için)
        affinity_loss = torch.sum(self.mse_loss(predicted_affinities, pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)
        
        return lig_interaction_loss, pro_lig_interaction_loss, affinity_loss

    @staticmethod
    def compute_metrics(predictions, targets):
        """
        Doğruluk (Accuracy) ve MAE metriklerini hesaplar.
        """
        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions
        if len(targets) == 6:
            mol_intearctions, pro_lig_interactions, pKs, pK_flags, _, _ = targets
        else:
            mol_intearctions, pro_lig_interactions, pKs, pK_flags, _ = targets
        device = predicted_affinities.device
        mol_intearctions     = create_var(mol_intearctions).to(device)
        pro_lig_interactions = create_var(pro_lig_interactions).to(device)
        pKs                  = create_var(pKs).to(device)
        pK_flags             = create_var(pK_flags).to(device)

        # Threshold 0.5 ile Binary Accuracy
        lig_interaction_result = torch.where(predicted_lig_interactions > 0.5, 1., 0.)
        lig_interaction_acc = torch.mean(torch.where(lig_interaction_result == mol_intearctions, 1., 0.)) 

        pro_lig_interaction_result = torch.where(predicted_interactions > 0.5, 1., 0.)
        pro_lig_interaction_acc = torch.mean(torch.where(pro_lig_interaction_result == pro_lig_interactions, 1., 0.))

        # Affinity MAE
        affinity_mae = torch.sum(torch.abs(predicted_affinities - pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)
        
        return lig_interaction_acc, pro_lig_interaction_acc, affinity_mae

    def load_parameters(self, parameters=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PLANET.param')):
        self.load_state_dict(torch.load(parameters, map_location=self.device))
        
    # --- Tarama (Screening) Yardımcı Fonksiyonları ---
    
    def cal_res_features_helper(self, fresidues, coordinates):
        fresidues = fresidues.to(self.device)
        coordinates = coordinates.to(self.device)
        fresidues = self.proteinegnn.pre_cal_res_features(fresidues, coordinates)
        return fresidues

    def cal_res_features(self, fresidues, batch_size):
        res_scope = [(i * int(fresidues.shape[0]), int(fresidues.shape[0])) for i in range(0, batch_size)]
        fresidues = fresidues.repeat(batch_size, 1)
        return fresidues, res_scope

    def screening(self, fresidues, res_scope, mol_feature_batch, res_coordinates):
        """
        Tarama modunda kullanılır. 
        DİKKAT: Artık res_coordinates (protein koordinatları) da parametre olarak verilmelidir.
        """
        # Mol batch unpack (Koordinat dahil)
        (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_feature_batch
        
        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)
        res_coordinates = create_var(res_coordinates).to(self.device)

        # Ligand Embedding
        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
        
        # Etkileşim Tahmini (Koordinatlı)
        _, _, predicted_affinities = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, res_coordinates, lig_coordinates
        )
        return predicted_affinities