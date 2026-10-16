import math, os, sys, torch
import torch.nn as nn

# add this directory to the path so the sibling modules import cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════
# Derived from PLANET's model module. The model class itself is unchanged from upstream;
# the only difference is the import line, which loads the layers module used by this arm.
# ═══════════════════════════════════════════════════════════════════════════
from layers_pliant_esm_pose import ProteinEGNN, LigandGAT, ProLig
from nnutils import create_var, create_var_gpu

class PLANET(nn.Module):
    def __init__(self, feature_dims, nheads, key_dims, value_dims, pro_update_inters, lig_update_iters, pro_lig_update_iters, device):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dims = feature_dims

        # 1. Katmanlar
        self.proteinegnn = ProteinEGNN(feature_dims, pro_update_inters, self.device)
        self.ligandgat = LigandGAT(feature_dims, nheads, key_dims, value_dims, lig_update_iters, self.device)

        # ProLig: now calls the version that expects coordinates, for the geometric computations
        self.prolig = ProLig(feature_dims, nheads, pro_lig_update_iters, self.device)

        # 2. Uncertainty weighting parameters
        self.log_sigma_lig = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_pl = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.log_sigma_affinity = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)

        # 3. Loss functions
        self.ligand_interaction_loss = nn.BCELoss(reduction='sum')
        self.pro_lig_interaction_loss = nn.BCELoss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, res_batch, mol_batch):
        (fresidues, res_map, res_scope, alpha_coordinates) = res_batch
        fresidues = fresidues.to(self.device)
        alpha_coordinates = create_var(alpha_coordinates).to(self.device)

        if len(mol_batch) == 6:
            (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coordinates) = mol_batch
        else:
            raise ValueError(f"Model 6 elemanlı mol_batch bekliyor (lig_coordinates dahil), ancak {len(mol_batch)} eleman geldi. Lütfen datautils dosyasını kontrol edin.")

        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        lig_coordinates = create_var(lig_coordinates).to(self.device)

        fresidues = self.proteinegnn(fresidues, alpha_coordinates, res_scope)
        fatoms = self.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)

        predicted_lig_interactions, predicted_interactions, predicted_affinities = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, alpha_coordinates, lig_coordinates
        )

        return (predicted_lig_interactions, predicted_interactions, predicted_affinities)

    def compute_loss(self, predictions, targets, res_batch, mol_batch):
        predicted_lig_interactions, predicted_interactions, predicted_affinities = predictions

        (_, _, res_scope, _) = res_batch
        lig_scope = mol_batch[4]

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
                # The decoy's CONTACT loss is MASKED here.
                # Reason: decoys are now placed INSIDE the pocket, so they really do have atoms
                # in contact. Telling the model "all contacts are 0" for them is a geometrically
                # WRONG label and pushes it towards a clash / impossibility shortcut. A decoy's
                # contribution to learning now comes from the affinity MARGIN rather than from the
                # contact label (margin_losses.molecule_margin_loss): y(binder) >= y(decoy) + m.
                pro_lig_interaction_loss.append(pl_loss)

            lig_interaction_count += lig_total_points
            pro_lig_interaction_count += pl_total_points

        lig_interaction_loss = torch.mean(torch.stack(lig_interaction_loss))
        pro_lig_interaction_loss = torch.mean(torch.stack(pro_lig_interaction_loss))

        affinity_loss = torch.sum(self.mse_loss(predicted_affinities, pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)

        return lig_interaction_loss, pro_lig_interaction_loss, affinity_loss

    @staticmethod
    def compute_metrics(predictions, targets):
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

        lig_interaction_result = torch.where(predicted_lig_interactions > 0.5, 1., 0.)
        lig_interaction_acc = torch.mean(torch.where(lig_interaction_result == mol_intearctions, 1., 0.))

        pro_lig_interaction_result = torch.where(predicted_interactions > 0.5, 1., 0.)
        pro_lig_interaction_acc = torch.mean(torch.where(pro_lig_interaction_result == pro_lig_interactions, 1., 0.))

        affinity_mae = torch.sum(torch.abs(predicted_affinities - pKs) * pK_flags) / (torch.sum(pK_flags) + 1e-6)

        return lig_interaction_acc, pro_lig_interaction_acc, affinity_mae

    def load_parameters(self, parameters=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PLANET.param')):
        self.load_state_dict(torch.load(parameters, map_location=self.device))

    # --- Screening helper functions ---

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

        _, _, predicted_affinities = self.prolig(
            fresidues, fatoms, res_scope, lig_scope, res_coordinates, lig_coordinates
        )
        return predicted_affinities
