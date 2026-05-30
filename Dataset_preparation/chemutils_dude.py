
print("DEBUG: chemutils_dude dosyası yükleniyor... (VERSION: FIXED_RESTORED)", flush=True)
import rdkit.Chem as Chem
import numpy as np
import torch
from collections import defaultdict

# --- SABITLER ---
MAX_NB = 10
periodic_table = Chem.GetPeriodicTable()
BLOSUM62 = {
    'ALA':np.array([[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0]],dtype=np.float32),
    'ARG':np.array([[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3]],dtype=np.float32),
    'ASN':np.array([[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3]],dtype=np.float32),
    'ASP':np.array([[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3]],dtype=np.float32),
    'CYS':np.array([[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]],dtype=np.float32),
    'GLN':np.array([[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2]],dtype=np.float32),
    'GLU':np.array([[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'GLY':np.array([[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3]],dtype=np.float32),
    'HIS':np.array([[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3]],dtype=np.float32),
    'ILE':np.array([[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3]],dtype=np.float32),
    'LEU':np.array([[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]],dtype=np.float32),
    'LYS':np.array([[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'MET':np.array([[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1]],dtype=np.float32),
    'PHE':np.array([[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1]],dtype=np.float32),
    'PRO':np.array([[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2]],dtype=np.float32),
    'SER':np.array([[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2]],dtype=np.float32),
    'THR':np.array([[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0]],dtype=np.float32),
    'TRP':np.array([[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3]],dtype=np.float32),
    'TYR':np.array([[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1]],dtype=np.float32),
    'VAL':np.array([[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]],dtype=np.float32)
}

PROTEIN_ELEMENTS = ['H','C','N','O','S','P','Se']
LIGADND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']
ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1 
BOND_FDIM = 6  

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGADND_ELEMENTS) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other'])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
        bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)

def near_pocket(pdb_line,centeroid):
    coordinates = pdb_line[30:55]
    x = float(coordinates[0:8])
    y = float(coordinates[8:16])
    z = float(coordinates[16:24])
    distance = np.sqrt(np.sum(np.square(np.array([x,y,z],dtype=np.float32)-centeroid)))
    return distance <= 12.0

def mass_center_from_pdb(pdb_lines):
    pdb_lines = [line for line in pdb_lines if line[76:78].strip().capitalize() in PROTEIN_ELEMENTS]
    if not pdb_lines: return np.zeros((1,3)), np.zeros((0,3)), np.zeros((0,3))

    atoms_mass = np.reshape(np.array([periodic_table.GetAtomicWeight(line[76:78].strip().capitalize()) for line in pdb_lines]),[-1,1])
    coordinates = np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pdb_lines],dtype=np.float32)
    
    if np.sum(atoms_mass) == 0: mass_center = np.mean(coordinates, axis=0, keepdims=True)
    else: mass_center = np.sum(coordinates * atoms_mass,axis=0,keepdims=True) / np.sum(atoms_mass)
    
    alpha_position = np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pdb_lines if line[12:16].strip()=="CA"],dtype=np.float32)
    
    # --- FIX: EGER CA ATOMU YOKSA MASS CENTER KULLAN ---
    if alpha_position.shape[0] == 0:
        alpha_position = mass_center
    # ---------------------------------------------------
    
    return mass_center, coordinates, alpha_position

class Residue():
    def __init__(self, residue_content, res_num_str, chain_id): 
        self.mass_center, self.coordinates, self.alpha_position = mass_center_from_pdb(residue_content)
        residue_type = residue_content[0][17:20]
        if residue_type == 'MSE': residue_type = 'MET'
        self.residue_type = residue_type
        self.feature = BLOSUM62.get(self.residue_type, np.zeros((1,20), dtype=np.float32))
    
    def get_feature(self): return self.feature
    def get_mass_center(self): return self.mass_center
    def get_alpha_position(self): return self.alpha_position

class ComplexPocket():
    def __init__(self,protein_pdb,ligand_sdf,pK=0,decoy_sdf=None):
        with open(protein_pdb,'r') as pdb_file:
            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')]
        
        self.ligand = Mol(ligand_sdf) if not isinstance(ligand_sdf, Mol) else ligand_sdf
        self.pK = pK
        
        centeroid = self.ligand.compute_centeroid()
        self.pocket_residues = []
        
        pocket_atoms_content = [line for line in pdb_content if near_pocket(line, centeroid)]
        res_dict = defaultdict(list)
        for line in pocket_atoms_content:
            res_id = (line[22:27].strip(), line[21])
            res_dict[res_id].append(line)
            
        for (res_num, chain), lines in res_dict.items():
            try: self.pocket_residues.append(Residue(lines, res_num, chain))
            except: continue
        self.res_count = len(self.pocket_residues)

class Mol():
    def __init__(self, mol):
        self.mol = mol
        self.atom_count = mol.GetNumAtoms()
    def get_atom_coordinates(self):
        conf = self.mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in range(self.atom_count)], dtype=np.float32)
    def compute_centeroid(self):
        return np.mean(self.get_atom_coordinates(), axis=0)
    def get_interaction_label(self):
        coords = self.get_atom_coordinates()
        dists = np.sqrt(np.sum((coords[:,None,:] - coords[None,:,:])**2, axis=-1))
        return np.where(dists < 4.0, 1.0, 0.0)

def tensorize_protein_pocket(protein_pockets):
    fresidues, res_scope, alpha_coordinates = [], [], []
    total_residues = 0
    res_map = [] 

    for pocket in protein_pockets:
        residue_count = len(pocket.pocket_residues)
        
        for residue in pocket.pocket_residues:
            fresidues.append(residue.get_feature())
            
            # Koordinat Al (Tamir edilmiş hali gelecek)
            pos = residue.get_alpha_position()
            
            # Yine de boşsa (ki olmamalı) Mass Center zorla
            if pos is None or len(pos) == 0 or (hasattr(pos, 'shape') and pos.shape[0] == 0):
                pos = residue.get_mass_center()
            
            # Boyut düzeltme
            if pos.ndim == 1: pos = pos.reshape(1, 3)
            
            alpha_coordinates.append(pos)
        
        res_scope.append((total_residues, residue_count))
        total_residues += residue_count

    if len(fresidues) > 0:
        fresidues = torch.from_numpy(np.concatenate(fresidues, axis=0).reshape(-1, 20))
        alpha_coordinates = torch.from_numpy(np.concatenate(alpha_coordinates, axis=0).astype(np.float32))
        res_map = torch.zeros((total_residues, total_residues, 1))
        for (start, count) in res_scope:
            res_map[start:start+count, start:start+count, :] = 1.0
    else:
        fresidues = torch.zeros((0, 20))
        alpha_coordinates = torch.zeros((0, 3))
        res_map = torch.zeros((0, 0, 1))

    return (fresidues, res_map, res_scope, alpha_coordinates)

def mol_batch_to_graph(mol_batch, auto_detect=True):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds = [], [padding] 
    in_bonds, all_bonds = [], [(-1,-1)] 
    lig_scope = []
    total_atoms = 0

    for mol in mol_batch:
        if auto_detect:
            try: Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
            except: pass
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() + total_atoms
            a2 = bond.GetEndAtomIdx() + total_atoms
            b = len(all_bonds)
            all_bonds.append((a1,a2))
            fbonds.append(torch.cat([fatoms[a1], bond_features(bond)], 0))
            in_bonds[a2].append(b)
            b = len(all_bonds)
            all_bonds.append((a2,a1))
            fbonds.append(torch.cat([fatoms[a2], bond_features(bond)], 0))
            in_bonds[a1].append(b)

        lig_scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    if len(fatoms) > 0: fatoms = torch.stack(fatoms, 0)
    else: fatoms = torch.zeros((0, ATOM_FDIM + BOND_FDIM))
    if len(fbonds) > 1: fbonds = torch.stack(fbonds, 0) 
    else: fbonds = torch.stack([padding], 0)

    total_bonds = len(all_bonds)
    agraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()
    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            if i < MAX_NB: agraph[a,i] = b
    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if i < MAX_NB and all_bonds[b2][0] != y: bgraph[b1,i] = b2

    return (fatoms, fbonds, agraph, bgraph, lig_scope)

def tensorize_ligand(mol_batch):
    graph = mol_batch_to_graph([m.mol for m in mol_batch])
    mol_int = torch.zeros(sum([m.atom_count for m in mol_batch]))
    return graph, mol_int

def tensorize_all(protein_pockets, decoy_flag=True):
    res_pack = tensorize_protein_pocket(protein_pockets)
    mol_pack, mol_int = tensorize_ligand([p.ligand for p in protein_pockets])
    # Dummy returns
    return res_pack, mol_pack, mol_int, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
