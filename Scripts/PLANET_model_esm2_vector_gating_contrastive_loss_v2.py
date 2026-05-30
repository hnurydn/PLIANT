import math, os, sys, torch
import torch.nn as nn
import torch.nn.functional as F

# Mevcut dizini path'e ekle (import sorunlarını önlemek için)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers_esm2_vector_gating import ProteinEGNN, LigandGAT, ProLig 
from nnutils import create_var, create_var_gpu

class PLANET(nn.Module):
    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dims = feature_dims
        
        # 1. Katmanlar (Vector + Gating + Dropout)
        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
        
        # ProLig: Vector Attention için koordinatları işleyen versiyon
        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device, dropout_rate=0.2)

        # 2. Uncertainty Weighting Parametreleri
        self.log_sigma_lig        = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_pl         = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_affinity   = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        # v2: Contrastive loss için ayrı sigma — pro-lig interaction skalasında bağımsız dengeleme
        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)

        # 3. Kayıp Fonksiyonları
        self.ligand_interaction_loss  = nn.BCELoss(reduction='sum')
        self.pro_lig_interaction_loss = nn.BCELoss(reduction='sum')
        self.mse_loss                 = nn.MSELoss(reduction='none')

    def forward(self, res_batch, mol_batch):
        """
        Modelin ileri yayılımı (Training/Validation sırasında çağrılır).
        res_batch: Protein verisi
        mol_batch: Ligand verisi (Koordinatlar dahil)
        """
        # --- A. Protein Verisini Hazırla ---
        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
        fresidues         = fresidues.to(self.device)
        alpha_coordinates = create_var(alpha_coordinates).to(self.device)

        # --- B. Ligand Verisini Hazırla ---
        if len(mol_batch) == 6:
            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
        else:
            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi.")

        fatoms          = fatoms.to(self.device)
        fbonds          = fbonds.to(self.device)
        agraph          = agraph.to(self.device)
        bgraph          = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)

        # --- C. Alt Katmanlardan Geçiş ---
        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
        fatoms    = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)

        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
        )
        
        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)
    
    def compute_loss(self, predictions, targets, res_batch, mol_batch):
        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions
        (_, _, res_scope, _) = res_batch
        lig_scope = mol_batch[4]

        # targets train'de 6 elemanlı (group_ids var), eval'da 5 elemanlı
        if len(targets) == 6:
            mol_intearctions, pro_lig_interactions, pKs, pK_flags, complex_labels, group_ids = targets
        else:
            mol_intearctions, pro_lig_interactions, pKs, pK_flags, complex_labels = targets
            group_ids = torch.zeros(len(complex_labels), dtype=torch.long, device=self.device)

        mol_intearctions     = create_var(mol_intearctions)
        pro_lig_interactions = create_var(pro_lig_interactions)
        pKs                  = create_var(pKs)
        pK_flags             = create_var(pK_flags)
        complex_labels       = create_var(complex_labels)
        group_ids            = group_ids.to(self.device)

        # ── 1. Etkileşim Kayıpları (değişmedi) ───────────────────────────────
        lig_interaction_count, pro_lig_interaction_count = 0, 0
        lig_interaction_loss, pro_lig_interaction_loss   = [], []

        for ((_, res_count), (_, atom_count), complex_label) in zip(res_scope, lig_scope, complex_labels):
            lig_total_points = atom_count ** 2
            l_loss = self.ligand_interaction_loss(
                predicted_lig_interactions[lig_interaction_count : lig_interaction_count + lig_total_points],
                mol_intearctions[lig_interaction_count : lig_interaction_count + lig_total_points]
            ) / lig_total_points

            pl_total_points = atom_count * res_count
            pl_loss = self.pro_lig_interaction_loss(
                predicted_interactions[pro_lig_interaction_count : pro_lig_interaction_count + pl_total_points],
                pro_lig_interactions[pro_lig_interaction_count : pro_lig_interaction_count + pl_total_points]
            ) / pl_total_points

            if complex_label == 1:
                lig_interaction_loss.append(l_loss)
            pro_lig_interaction_loss.append(pl_loss)

            lig_interaction_count     += lig_total_points
            pro_lig_interaction_count += pl_total_points

        lig_interaction_loss     = torch.mean(torch.stack(lig_interaction_loss))
        pro_lig_interaction_loss = torch.mean(torch.stack(pro_lig_interaction_loss))

        # ── 2. Affinity Loss (değişmedi) ─────────────────────────────────────
        affinity_mse = torch.sum(
            self.mse_loss(predicted_affinities, pKs) * pK_flags
        ) / (torch.sum(pK_flags) + 1e-6)

        # ── 3. Contact-tabanlı Contrastive Loss (v3) ─────────────────────────
        # Her molekül için predicted_interactions (atom×residue matrisi) üzerinde
        # softmax-weighted mean alınır — affinity hesabındaki ağırlıklı toplam
        # mantığıyla tutarlı.
        #
        # Neden softmax-weighted mean?
        #   - Düz mean: yüksek contact skorları düşük skorların arasında boğulur
        #     (1000 çiftin ortalaması ≈ 0.05, signal kaybolur)
        #   - Softmax-weighted: yüksek skorlu çiftler daha fazla ağırlık alır,
        #     binder/decoy ayrımı netleşir — affinity'deki interaction_weight ile aynı fikir
        #
        # margin=0.1: softmax-weighted değerler 0-1 arasında ama daha yüksek
        #   seviyelerde olacak, 0.1'lik fark makul bir hedef.

        contact_per_mol = []
        ptr = 0
        for (_, res_count), (_, atom_count) in zip(res_scope, lig_scope):
            n      = atom_count * res_count
            scores = predicted_interactions[ptr : ptr + n]          # (n,) 0-1 arası
            weights = torch.softmax(scores, dim=0)                   # yüksek skora daha fazla ağırlık
            mol_contact = (scores * weights).sum()                   # softmax-weighted sum ≡ weighted mean
            contact_per_mol.append(mol_contact)
            ptr += n
        contact_per_mol = torch.stack(contact_per_mol)  # (N_mol,) — her mol için tek skaler

        contrastive_loss = torch.tensor(0.0, device=self.device)
        group_losses     = []

        for g in torch.unique(group_ids):
            g_mask        = (group_ids == g)
            g_active_mask = g_mask & (complex_labels == 1) & (pK_flags == 1)
            g_decoy_mask  = g_mask & (complex_labels == 0)

            g_active_contact = contact_per_mol[g_active_mask]  # binder contact ortalaması
            g_decoy_contact  = contact_per_mol[g_decoy_mask]   # decoy contact ortalaması

            if len(g_active_contact) > 0 and len(g_decoy_contact) > 0:
                # diff > 0: decoy > binder (yanlış) → penalize et
                # diff < 0: binder > decoy (doğru) → küçük loss
                diff   = g_decoy_contact.unsqueeze(1) - g_active_contact.unsqueeze(0)
                margin = 0.0  # önce sadece aggregation değişikliğinin etkisini izle; Con_L 0.69'da kalırsa 0.1'e çıkar
                group_losses.append(torch.mean(F.softplus(diff - margin)))

        if group_losses:
            contrastive_loss = torch.mean(torch.stack(group_losses))

        return lig_interaction_loss, pro_lig_interaction_loss, affinity_mse, contrastive_loss

    @staticmethod
    def compute_metrics(predictions, targets):
        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions
        mol_intearctions, pro_lig_interactions, pKs, pK_flags = targets[0], targets[1], targets[2], targets[3]
        _ = targets[4]  # complex_labels

        mol_intearctions     = create_var(mol_intearctions)
        pro_lig_interactions = create_var(pro_lig_interactions)
        pKs                  = create_var(pKs)
        pK_flags             = create_var(pK_flags)

        lig_interaction_result = torch.where(predicted_lig_interactions > 0.5, 1., 0.)
        lig_interaction_acc    = torch.mean(torch.where(lig_interaction_result == mol_intearctions, 1., 0.))

        pro_lig_interaction_result = torch.where(predicted_interactions > 0.5, 1., 0.)
        pro_lig_interaction_acc    = torch.mean(torch.where(pro_lig_interaction_result == pro_lig_interactions, 1., 0.))

        affinity_mae = torch.sum(torch.abs(predicted_affinities - pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)
        
        return lig_interaction_acc, pro_lig_interaction_acc, affinity_mae

    def load_parameters(self, parameters=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PLANET.param')):
        self.load_state_dict(torch.load(parameters, map_location=self.device))

    # ── Tarama (Screening) Yardımcı Fonksiyonları ────────────────────────────

    def cal_res_features_helper(self, fresidues, coordinates):
        fresidues   = fresidues.to(self.device)
        coordinates = coordinates.to(self.device)
        fresidues   = self.proteinegnn.pre_cal_res_features(fresidues, coordinates)
        return fresidues

    def cal_res_features(self, fresidues, batch_size):
        res_scope = [(i * int(fresidues.shape[0]), int(fresidues.shape[0])) for i in range(0, batch_size)]
        fresidues = fresidues.repeat(batch_size, 1)
        return fresidues, res_scope

    def screening(self, fresidues, res_scope, mol_feature_batch, res_coordinates):
        (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_feature_batch
        
        fatoms          = fatoms.to(self.device)
        fbonds          = fbonds.to(self.device)
        agraph          = agraph.to(self.device)
        bgraph          = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)
        res_coordinates = create_var(res_coordinates).to(self.device)

        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
        
        _, _, predicted_affinities = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, res_coordinates, lig_coordinates
        )
        return predicted_affinities