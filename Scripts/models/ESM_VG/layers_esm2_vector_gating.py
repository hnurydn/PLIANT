import torch
import torch.nn as nn
import torch.nn.functional as F
import math, sys
from nnutils import create_var
from nnutils import index_select_ND

# --- DEĞİŞİKLİK YOK (ProteinGAT) ---
class ProteinGAT(nn.Module):
    def __init__(self,feature_dims,nheads,key_dims,value_dims,update_iters,device):
        super().__init__()
        self.nheads = nheads
        self.feature_dims = feature_dims
        self.key_dims = key_dims
        self.value_dims = value_dims
        self.update_iters = update_iters
        self.res_embedding = nn.Sequential(nn.Linear(512,feature_dims,bias=False),nn.LeakyReLU(0.1))
        self.linear_Q = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_K = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_V = nn.Linear(feature_dims,nheads*value_dims,bias=False)
        self.linear_out = nn.Sequential(nn.Linear(value_dims,feature_dims),nn.LeakyReLU(0.1))
        self.linear_last = nn.Sequential(nn.Linear(2*feature_dims,feature_dims),nn.LeakyReLU(0.1))
        self.device = device

    def split(self,q,k,v):
        q = torch.reshape(q,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,0,2]) #[h,n,k]
        k = torch.reshape(k,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,2,0]) #[h,k,n]
        v = torch.reshape(v,shape=[self.node_size,self.nheads,self.value_dims]).permute([1,0,2]) #[h,n,v]
        return q,k,v
    
    def forward(self,fresidues,distance_weight,res_mask):
        fresidues = self.res_embedding(fresidues)
        fresidues =  fresidues + torch.sum(torch.unsqueeze(fresidues,0) * distance_weight,dim=1)
        initial_features = fresidues
        self.node_size = fresidues.shape[0]
        for _ in range(self.update_iters):
            q = self.linear_Q(fresidues)
            k = self.linear_K(fresidues)
            v = self.linear_V(fresidues)
            q,k,v = self.split(q,k,v)
            attention_matrix = torch.matmul(q,k)/ math.sqrt(self.key_dims) 
            attention_matrix = attention_matrix + res_mask #[h,n,n]
            attention_matrix = torch.softmax(attention_matrix,dim=-1) 
            v = torch.matmul(attention_matrix,v).permute([1,0,2]) #[h,n,f] -- [n,h,f]
            fresidues = self.linear_out(torch.sum(v,dim=1)) #[n,f]
        fresidues = self.linear_last(torch.cat([fresidues,initial_features],dim=-1))
        return fresidues

# --- DEĞİŞİKLİK YOK (LigandGAT) ---
class LigandGAT(nn.Module):
    def __init__(self,feature_dims,nheads,key_dims,value_dims,update_iters,device):
        super().__init__()
        self.nheads = nheads
        self.feature_dims = feature_dims
        self.key_dims = key_dims
        self.value_dims = value_dims
        self.update_iters = update_iters
        self.device = device
        
        self.fbonds_embedding = nn.Linear(31+6,feature_dims)
        self.linear_Q = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_K = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_V = nn.Linear(feature_dims,nheads*value_dims,bias=False)
        self.fbonds_out = nn.Linear(value_dims,feature_dims,bias=False)
        self.linear_last = nn.Sequential(nn.Linear(31+feature_dims,feature_dims),nn.LeakyReLU(0.1))

    def split(self,q,k,v):
        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3]) 
        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])
        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])
        return q,k,v

    def forward(self,fatoms,fbonds,agraph,bgraph,lig_scope):
        self.bonds_size = fbonds.shape[0]
        self.nei_size = bgraph.shape[1]
        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        fbonds_input = self.fbonds_embedding(fbonds)
        message = F.leaky_relu(fbonds_input,negative_slope=0.1)
        
        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)
        for _ in range(self.update_iters-1):
            nei_message = index_select_ND(message,0,bgraph)
            q = self.linear_Q(message)  
            k = self.linear_K(nei_message)
            v = self.linear_V(nei_message)
            q,k,v = self.split(q,k,v)
            attention_matrix = torch.matmul(q,k)/ math.sqrt(self.key_dims) + nei_mask
            attention_matrix = torch.softmax(attention_matrix,dim=-1)
            nei_message = torch.matmul(attention_matrix,v).permute([0,2,1,3])
            nei_message = self.fbonds_out(torch.mean(nei_message,dim=2)).reshape([self.bonds_size,-1])
            message = F.leaky_relu(fbonds_input + nei_message,negative_slope=0.1)
        
        nei_message = torch.sum(index_select_ND(message,0,agraph),dim=1)
        fatoms = self.linear_last(torch.cat([fatoms,nei_message],dim=-1))
        
        return fatoms

# --- GaussianSmearing (Vector için gerekli) ---
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=20.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / num_gaussians) ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))

# --- [GÜNCELLENDİ] ProteinLigandAttention (+Dropout) ---
class ProteinLigandAttention(nn.Module):
    def __init__(self, feature_dims, nheads, update_iters, device, dropout_rate=0.2):
        super().__init__()
        self.feature_dims = feature_dims
        self.nheads = nheads
        self.update_iters = update_iters
        self.device = device
        
        self.lig_transform = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))
        self.res_transform = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))

        self.lig_Q = nn.Linear(feature_dims, feature_dims, bias=False)
        self.lig_K = nn.Linear(feature_dims, feature_dims, bias=False)
        self.lig_V = nn.Linear(feature_dims, feature_dims, bias=False)
        self.lig_out = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))

        self.res_Q = nn.Linear(feature_dims, feature_dims, bias=False)
        self.res_K = nn.Linear(feature_dims, feature_dims, bias=False)
        self.res_V = nn.Linear(feature_dims, feature_dims, bias=False)
        self.res_out = nn.Sequential(nn.Linear(feature_dims, feature_dims, bias=False), nn.LeakyReLU(0.1))

        self.num_rbf = 32
        self.rbf_fn = GaussianSmearing(start=0.0, stop=15.0, num_gaussians=self.num_rbf)
        
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.num_rbf + 3, feature_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dims, 1) 
        )

    def forward(self, fatoms, fresidues, lig_scope, res_scope, res_coords, lig_coords):
        fatoms = self.lig_transform(fatoms)
        fresidues = self.res_transform(fresidues)

        for _ in range(self.update_iters):
            fresidues_new, fatoms_new = [], []
            
            for ((start_res, res_count), (start_atom, atom_count)) in zip(res_scope, lig_scope):
                fatoms_batch = fatoms[start_atom:start_atom+atom_count]
                fresidues_batch = fresidues[start_res:start_res+res_count]
                
                l_coords = lig_coords[start_atom:start_atom+atom_count]
                r_coords = res_coords[start_res:start_res+res_count]

                atoms_q = self.lig_Q(fatoms_batch) 
                atom_k = self.lig_K(fatoms_batch)
                atom_v = self.lig_V(fatoms_batch)
                
                res_q = self.res_Q(fresidues_batch)
                res_k = self.res_K(fresidues_batch)
                res_v = self.res_V(fresidues_batch)

                diff_vector = l_coords.unsqueeze(1) - r_coords.unsqueeze(0) # [Atom, Res, 3]
                dist = torch.norm(diff_vector, dim=-1) + 1e-6
                direction = diff_vector / dist.unsqueeze(-1)
                
                rbf_feat = self.rbf_fn(dist).to(self.device)
                geo_input = torch.cat([rbf_feat, direction], dim=-1)
                geo_bias = self.geo_mlp(geo_input).squeeze(-1) 

                attn_lig_to_pro = torch.matmul(atoms_q, res_k.transpose(0, 1))
                attn_weights_lp = torch.sigmoid(attn_lig_to_pro + geo_bias) 
                
                attn_pro_to_lig = torch.sigmoid(torch.matmul(res_q, atom_k.transpose(0, 1)) + geo_bias.transpose(0, 1))

                update_lig = torch.matmul(attn_weights_lp, res_v)
                fatoms_batch = self.lig_out(update_lig) / math.sqrt(res_count) + fatoms_batch
                
                update_pro = torch.matmul(attn_pro_to_lig, atom_v)
                fresidues_batch = self.res_out(update_pro) / math.sqrt(atom_count) + fresidues_batch

                fatoms_new.append(fatoms_batch)
                fresidues_new.append(fresidues_batch)
            
            fatoms = torch.cat(fatoms_new, dim=0)
            fresidues = torch.cat(fresidues_new, dim=0)
            
        return fatoms, fresidues

# --- [GÜNCELLENDİ] ProLig (+Dropout, Affinity Normalizasyonu) ---
class ProLig(nn.Module):
    def __init__(self, feature_dims, nheads, update_iters, device, dropout_rate=0.2):
        super().__init__()
        self.feature_dims = feature_dims
        self.device = device
        self.prolig_attention = ProteinLigandAttention(feature_dims, nheads, update_iters, device, dropout_rate)

        self.linear_lig_interaction = nn.Sequential(
            nn.Linear(feature_dims,feature_dims,bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.linear_lig_interaction_last = nn.Sequential(
            nn.Linear(feature_dims,1,bias=False),
            nn.Sigmoid(),
        )
        
        self.linear_pocket_interaction = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.linear_ligand_interaction = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        self.pro_lig_interaction = nn.Sequential(
            nn.Linear(feature_dims,1,bias=False),
            nn.Sigmoid(),
        )
        
        self.linear_pocket_affinity = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        self.linear_ligand_affinity = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.linear_affinity = nn.Sequential(
            nn.Linear(feature_dims, feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims, 1, bias=False) 
        )

    def forward(self,fresidues,fatoms,res_scope,lig_scope, res_coords, lig_coords):
        fatoms,fresidues = self.prolig_attention(fatoms,fresidues,lig_scope,res_scope, res_coords, lig_coords)
        
        predicted_lig_interactions,predicted_interactions,predicted_affinities = [],[],[]
        for ((start_res,res_count),(start_atom,atom_count)) in zip(res_scope,lig_scope):
            
            pocket_features = torch.unsqueeze(fresidues[start_res:start_res+res_count],dim=0) 
            ligand_features = torch.unsqueeze(fatoms[start_atom:start_atom+atom_count],dim=1) 
            
            predicted_lig_interaction = self.linear_lig_interaction_last(torch.multiply(
                self.linear_lig_interaction(ligand_features).unsqueeze(1), \
                self.linear_lig_interaction(ligand_features).unsqueeze(0)
                ))

            predicted_lig_interactions.append(predicted_lig_interaction.reshape([-1]))

            predicted_interaction = self.pro_lig_interaction(torch.multiply(
                self.linear_ligand_interaction(ligand_features),
                self.linear_pocket_interaction(pocket_features)
                )) 
            predicted_interactions.append(torch.reshape(predicted_interaction,shape=[-1]))

            complex_features = torch.multiply(
                self.linear_ligand_affinity(ligand_features),
                self.linear_pocket_affinity(pocket_features)
            )

            # [DÜZELTİLDİ] Normalize edilmiş interaction weight ile affinity hesabı
            interaction_weight = predicted_interaction / (predicted_interaction.sum() + 1e-6)
            predict_affinity = torch.sum(
                self.linear_affinity(complex_features) * interaction_weight
            ).reshape([1])

            predicted_affinities.append(predict_affinity)
        
        predicted_lig_interactions = torch.cat(predicted_lig_interactions)
        predicted_interactions = torch.cat(predicted_interactions)
        predicted_affinities = torch.cat(predicted_affinities)
        return predicted_lig_interactions,predicted_interactions,predicted_affinities
    
# --- [GÜNCELLENDİ] ProteinEGNN (CROSS-MODAL GATING EKLENDİ) ---
class ProteinEGNN(nn.Module):
    def __init__(self, feature_dims, update_iters, device):
        super().__init__()
        self.update_iters = update_iters
        self.feature_dims = feature_dims
        self.device = device
        
        # ESM embedding layer (LayerNorm ile normalize et)
        self.res_embedding = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, feature_dims, bias=False),
            nn.LeakyReLU(0.1)
        )
        
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dims * 2, feature_dims),
            nn.Sigmoid() 
        )
        
        self.linear_e = nn.Sequential(
            nn.Linear(2*feature_dims+1, feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims, feature_dims),
            nn.LeakyReLU(0.1)
        )
        self.linear_inf = nn.Sequential(
            nn.Linear(feature_dims, 1),
            nn.Sigmoid()
        )
        self.linear_h = nn.Sequential(
            nn.Linear(2*feature_dims, feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims, feature_dims)
        )

    def forward(self, fresidues, coordinates, res_scope):
        esm_features = self.res_embedding(fresidues)
        structural_features = esm_features.clone()
        
        for _ in range(self.update_iters):
            fresidues_new = []
            
            for (start_res, res_count) in res_scope:
                struct_batch = structural_features[start_res:start_res+res_count]
                esm_batch = esm_features[start_res:start_res+res_count]
                coords_batch = coordinates[start_res:start_res+res_count]
                
                broadcast_shape = [struct_batch.shape[0], struct_batch.shape[0], self.feature_dims]
                coor_diff = coords_batch.unsqueeze(1) - coords_batch.unsqueeze(0)
                
                m_input = torch.cat([
                    struct_batch.unsqueeze(1).broadcast_to(broadcast_shape),
                    struct_batch.unsqueeze(0).broadcast_to(broadcast_shape),
                    torch.sum(torch.square(coor_diff), dim=-1, keepdim=True)
                ], dim=-1)
                
                m = self.linear_e(m_input)
                e = self.linear_inf(m)
                
                mask = torch.ones([res_count, res_count]) - torch.eye(res_count)
                mask = mask.unsqueeze(-1).to(self.device)
                e = e * mask
                m_pre = torch.sum(m*e, dim=1)
                
                struct_update = self.linear_h(torch.cat([struct_batch, m_pre], dim=-1)) + struct_batch
                
                gate_input = torch.cat([struct_update, esm_batch], dim=-1)
                z = self.gating_network(gate_input) 
                
                fused_features = (z * struct_update) + ((1 - z) * esm_batch)
                
                fresidues_new.append(fused_features)
                
            structural_features = torch.cat(fresidues_new, dim=0)
            
        return structural_features

    def pre_cal_res_features(self, fresidues, coordinates):
        esm_features = self.res_embedding(fresidues)
        structural_features = esm_features.clone()
        
        for _ in range(self.update_iters):
            struct_batch = structural_features
            esm_batch = esm_features
            
            broadcast_shape = [struct_batch.shape[0], struct_batch.shape[0], self.feature_dims]
            coor_diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
            
            m_input = torch.cat([
                struct_batch.unsqueeze(1).broadcast_to(broadcast_shape),
                struct_batch.unsqueeze(0).broadcast_to(broadcast_shape),
                torch.sum(torch.square(coor_diff), dim=-1, keepdim=True)
            ], dim=-1)
            
            m = self.linear_e(m_input)
            e = self.linear_inf(m)
            mask = torch.ones([struct_batch.shape[0], struct_batch.shape[0]]) - torch.eye(struct_batch.shape[0])
            mask = mask.unsqueeze(-1).to(self.device)
            e = e * mask
            m_pre = torch.sum(m*e, dim=1)
            
            struct_update = self.linear_h(torch.cat([struct_batch, m_pre], dim=-1)) + struct_batch
            
            gate_input = torch.cat([struct_update, esm_batch], dim=-1)
            z = self.gating_network(gate_input)
            structural_features = (z * struct_update) + ((1 - z) * esm_batch)
            
        return structural_features