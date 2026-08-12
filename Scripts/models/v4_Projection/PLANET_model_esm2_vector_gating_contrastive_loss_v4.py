import math, os, sys, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers_esm2_vector_gating_projection_head import ProteinEGNN, LigandGAT, ProLig
from nnutils import create_var, create_var_gpu

class PLANET(nn.Module):
    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dims = feature_dims

        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)
        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device, dropout_rate=0.2)

        # Uncertainty Weighting parametreleri
        self.log_sigma_lig         = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_pl          = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_affinity    = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        # v4: Contrastive için ayrı sigma — latent uzayda bağımsız sinyal
        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)

        self.ligand_interaction_loss  = nn.BCELoss(reduction='sum')
        self.pro_lig_interaction_loss = nn.BCELoss(reduction='sum')
        self.mse_loss                 = nn.MSELoss(reduction='none')

    def forward(self, res_batch, mol_batch):
        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
        fresidues = fresidues.to(self.device)
        alpha_coordinates = create_var(alpha_coordinates).to(self.device)

        if len(mol_batch) == 6:
            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
        else:
            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor, ancak {len(mol_batch)} eleman geldi.")

        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)

        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
        fatoms    = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)

        # v4: 4 çıktı (ek olarak z_complexes)
        predicted_lig_interactions, predicted_interactions, predicted_affinities, z_complexes = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
        )

        return (predicted_lig_interactions, predicted_interactions, predicted_affinities, z_complexes)

    def compute_loss(self, predictions, targets, res_batch, mol_batch):
        # v4: 4 çıktı unpack
        predicted_lig_interactions, predicted_interactions, predicted_affinities, z_complexes = predictions
        (_, _, res_scope, _) = res_batch
        lig_scope = mol_batch[4]

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

        # 1. Etkileşim Kayıpları
        lig_interaction_count, pro_lig_interaction_count = 0, 0
        lig_interaction_loss, pro_lig_interaction_loss = [], []

        for ((_, res_count), (_, atom_count), complex_label) in zip(res_scope, lig_scope, complex_labels):
            lig_total_points = atom_count**2
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

        # 2. Affinity Loss (MSE)
        affinity_mse = torch.sum(self.mse_loss(predicted_affinities, pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)

        # 3. Latent-space Contrastive Loss (v4)
        # z_complexes: (N_mol, 128) — her molekül için L2-normalized projection embedding
        # Her group içinde: binder ile decoy embedding'lerinin cosine similarity'si düşük olsun
        contrastive_loss = torch.tensor(0.0, device=self.device)
        group_losses = []
        margin = 0.0  # cosine similarity için makul eşik (-1..1 aralığında)
                     # binder-decoy cosine sim < 0 olmalı (zıt yönde olsun)

        for g in torch.unique(group_ids):
            g_mask        = (group_ids == g)
            g_active_mask = g_mask & (complex_labels == 1) & (pK_flags == 1)
            g_decoy_mask  = g_mask & (complex_labels == 0)

            z_active = z_complexes[g_active_mask]  # (n_a, 128)
            z_decoys = z_complexes[g_decoy_mask]   # (n_d, 128)

            if len(z_active) > 0 and len(z_decoys) > 0:
                # Cosine similarity (z_active ve z_decoys zaten L2-normalized)
                sim_matrix = z_active @ z_decoys.T  # (n_a, n_d), -1..1 arası
                # margin = 0 ise: sim_matrix < 0 olmalı (cosine zıt)
                # softplus(sim - margin): sim büyükse penalty büyük
                group_losses.append(torch.mean(F.softplus(sim_matrix - margin)))

        if group_losses:
            contrastive_loss = torch.mean(torch.stack(group_losses))

        return lig_interaction_loss, pro_lig_interaction_loss, affinity_mse, contrastive_loss

    @staticmethod
    def compute_metrics(predictions, targets):
        # v4: 4 çıktı, ilk üçü v1 ile aynı
        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions[0], predictions[1], predictions[2]
        mol_intearctions, pro_lig_interactions, pKs, pK_flags = targets[0], targets[1], targets[2], targets[3]
        _ = targets[4]

        mol_intearctions     = create_var(mol_intearctions)
        pro_lig_interactions = create_var(pro_lig_interactions)
        pKs                  = create_var(pKs)
        pK_flags             = create_var(pK_flags)

        lig_interaction_result = torch.where(predicted_lig_interactions > 0.5, 1., 0.)
        lig_interaction_acc = torch.mean(torch.where(lig_interaction_result == mol_intearctions, 1., 0.))

        pro_lig_interaction_result = torch.where(predicted_interactions > 0.5, 1., 0.)
        pro_lig_interaction_acc = torch.mean(torch.where(pro_lig_interaction_result == pro_lig_interactions, 1., 0.))

        affinity_mae = torch.sum(torch.abs(predicted_affinities - pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)

        return lig_interaction_acc, pro_lig_interaction_acc, affinity_mae

    def load_parameters(self, parameters=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PLANET.param')):
        self.load_state_dict(torch.load(parameters, map_location=self.device))

    # Tarama (screening) yardımcıları — v1 ile aynı
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
        (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_feature_batch
        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)
        res_coordinates = create_var(res_coordinates).to(self.device)

        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)

        # v4: 4 çıktı, screening için sadece affinity lazım
        _, _, predicted_affinities, _ = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, res_coordinates, lig_coordinates
        )
        return predicted_affinities