# layers_ligand_lm_gate.py
# PLANET+ligand-LM uzantisi icin paylasilan, mimari-bagimsiz gate modulu.
# Tasarim (2026-08-06 planindaki "Versiyon B" MVP): LigandGAT atom-seviyesi
# islemi TAMAMLADIKTAN SONRA, molekul-basi tek ChemBERTa vektorunu o
# molekulun TUM atomlarina broadcast edip, ProteinEGNN'deki Cross-Modal
# Gating (CMG) ile AYNI desende (z=sigmoid(Linear([x,y])), fused=z*x+(1-z)*y)
# atom temsiliyle karistirir. Protein tarafindaki asimetrik fark: ESM-2
# rezidu-basina zaten farkli bir vektorken, ChemBERTa molekul-basina TEK
# vektor oldugu icin broadcast + per-atom ogrenilen gate agirligi kullanilir.
#
# Bu dosya HERHANGI bir PLANET_model_*.py tarafindan import edilebilir,
# var olan layers dosyalarina (layers.py / layers_esm2_vector_gating*.py)
# dokunmadan.
import torch
import torch.nn as nn


class LigandLMGate(nn.Module):
    def __init__(self, feature_dims, lm_dim=768, device='cpu'):
        super().__init__()
        self.device = device
        self.feature_dims = feature_dims

        # ChemBERTa (768) -> feature_dims projeksiyonu (LayerNorm ile normalize,
        # ProteinEGNN'in guncel res_embedding deseniyle tutarli)
        self.lm_proj = nn.Sequential(
            nn.LayerNorm(lm_dim),
            nn.Linear(lm_dim, feature_dims, bias=False),
            nn.LeakyReLU(0.1),
        )

        # CMG-tarzi gate agi: girdi [atom_feature, broadcast_lm_feature]
        self.gate = nn.Sequential(
            nn.Linear(feature_dims * 2, feature_dims),
            nn.Sigmoid(),
        )

    def forward(self, fatoms, lig_scope, lig_lm_features):
        """
        fatoms: [total_atoms_in_batch, feature_dims] -- LigandGAT ciktisi
        lig_scope: [(start_atom, atom_count), ...] -- batch'teki her ligandin
                   atom araligi (PLANET'in standart scope formati)
        lig_lm_features: [n_ligands_in_batch, lm_dim] -- her ligandin
                   ChemBERTa (pooler_output) vektoru

        Donus: [total_atoms_in_batch, feature_dims] -- gate ile karistirilmis
               fatoms, boyutu/sirasi girdiyle birebir ayni (ProLig'e direkt
               verilebilir).
        """
        assert len(lig_scope) == lig_lm_features.shape[0], (
            f"lig_scope uzunlugu ({len(lig_scope)}) ile lig_lm_features "
            f"({lig_lm_features.shape[0]}) uyusmuyor"
        )
        lm_proj = self.lm_proj(lig_lm_features)  # [n_ligands, feature_dims]

        fatoms_new = []
        for i, (start, count) in enumerate(lig_scope):
            atom_feats = fatoms[start:start + count]              # [count, F]
            lm_vec = lm_proj[i].unsqueeze(0).expand(count, -1)      # [count, F]
            gate_input = torch.cat([atom_feats, lm_vec], dim=-1)    # [count, 2F]
            z = self.gate(gate_input)                               # [count, F]
            fused = z * atom_feats + (1 - z) * lm_vec
            fatoms_new.append(fused)

        return torch.cat(fatoms_new, dim=0)
