"""
PLANET_datautils_esm2.py
=========================
lmdb_planet_v2 ile çalışır.

Her binder key'i → ComplexPocket(ligand=gerçek, decoys=[Mol1, Mol2, ...])

Train modu:
- Her batch = 1 binder + rastgele 11 decoy (aynı pocket'tan)
- 11'den az decoy'u olan pocket'lar atlanır

Eval modu (decoy_count verilince):
- Her binder için ilk min(decoy_count, mevcut) kadar decoy alınır
- Binder + decoylar deterministik olarak işlenir
"""

import os, pickle, random, lmdb, json
import numpy as np
import torch
from torch.utils.data import Dataset
from chemutils_esm2 import tensorize_protein_pocket, tensorize_ligand


class ProLigDataset(Dataset):

    def __init__(self, lmdb_path, batch_size, shuffle=True,
                 decoy_flag=True, binder_per_batch=1, decoy_count=None):

        self.lmdb_path        = lmdb_path
        self.batch_size       = batch_size
        self.decoy_flag       = decoy_flag
        self.binder_per_batch = binder_per_batch
        self.decoy_count      = decoy_count

        self._env = None
        self._txn = None
        self._pid = None

        meta_path = lmdb_path.rstrip('/\\')
        if meta_path.endswith('.lmdb'):
            meta_path = meta_path[:-5]
        meta_path += '_meta.json'

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.binders = meta

        if decoy_count is not None:
            self.batches = self._build_eval_batches()
            mode = 'eval'
        else:
            self.batches = self._build_train_batches(shuffle)
            mode = 'train'

        total_decoys = sum(m.get('n_decoys', 0) for m in self.binders)
        print(f'✅ LMDB v2 Dataset hazır [{mode}]: {len(self.batches)} batch '
              f'({len(self.binders)} binder, '
              f'ort. {total_decoys/max(len(self.binders),1):.1f} decoy/binder)')

    def _build_train_batches(self, shuffle=True):
        # Her batch = 1 binder (kendi 11 decoyuyla işlenecek)
        binders = self.binders.copy()
        if shuffle:
            random.shuffle(binders)
        return [[b] for b in binders]

    def _build_eval_batches(self):
        return [[b] for b in self.binders]

    def reshuffle(self):
        if self.decoy_count is None:
            self.batches = self._build_train_batches(shuffle=True)

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
            self._txn = self._env.begin(write=False, buffers=True)
            self._pid = pid
        return self._env

    def _load_pocket(self, key):
        self._get_env()
        raw = self._txn.get(key.encode())
        if raw is None:
            return None
        return pickle.loads(bytes(raw))

    def __len__(self):
        return len(self.batches)

    def _tensorize_train(self, idx):
        """
        Train modu: 1 binder + max 11 decoy (aynı pocket'tan).
        11'den az decoy'u olan pocket'lar atlanır.
        """
        record = self.batches[idx][0]
        try:
            obj = self._load_pocket(record['key'])
            if obj is None or not hasattr(obj, 'pocket_residues'):
                return None
            obj.pK = float(record['pK'])
        except Exception:
            return None

        # 11'den az decoy varsa atla
        n_decoys_available = len(obj.decoys) if obj.decoys else 0
        if n_decoys_available < 11:
            return None

        # Protein özelliklerini tensorize et
        res_f = tensorize_protein_pocket([obj])

        # Binder
        mol_f_binder, mol_int_binder = tensorize_ligand([obj.ligand])
        pl_int_binder = torch.from_numpy(
            obj.pro_lig_interaction.reshape([-1]).astype(np.float32))

        all_mol_f   = [mol_f_binder]
        all_mol_int = [mol_int_binder]
        all_pl_int  = [pl_int_binder]
        all_pKs     = [torch.tensor([obj.pK], dtype=torch.float32)]
        all_flags   = [torch.tensor([1.0])]
        all_labels  = [torch.tensor([1.0])]

        # 11 decoy — rastgele seç
        selected_indices = np.random.choice(n_decoys_available, size=11, replace=False)
        for idx_d in selected_indices:
            try:
                decoy_mol = obj.decoys[idx_d]
                mol_f_d, mol_int_d = tensorize_ligand([decoy_mol])
                pl_int_d = torch.from_numpy(
                    np.zeros([decoy_mol.atom_count, obj.res_count],
                             dtype=np.float32).reshape([-1]))
                all_mol_f.append(mol_f_d)
                all_mol_int.append(mol_int_d)
                all_pl_int.append(pl_int_d)
                all_pKs.append(torch.tensor([0.0]))
                all_flags.append(torch.tensor([0.0]))
                all_labels.append(torch.tensor([0.0]))
            except Exception:
                continue

        # mol_f birleştir
        merged = []
        for i in range(len(mol_f_binder)):
            if isinstance(mol_f_binder[i], list):
                fixed_scope = []
                atom_offset = 0
                for mf in all_mol_f:
                    for (s, l) in mf[i]:
                        fixed_scope.append((atom_offset, l))
                        atom_offset += l
                merged.append(fixed_scope)
            else:
                merged.append(torch.cat([mf[i] for mf in all_mol_f], dim=0))
        mol_f_combined = tuple(merged)

        # res_scope: aynı protein n_mol kez tekrar
        n_mol = len(all_mol_f)
        fresidues, res_map, res_scope, alpha_coords = res_f
        res_scope_expanded = res_scope * n_mol
        res_f_expanded = (fresidues, res_map, res_scope_expanded, alpha_coords)

        targets = (
            torch.cat(all_mol_int),
            torch.cat(all_pl_int),
            torch.cat(all_pKs),
            torch.cat(all_flags),
            torch.cat(all_labels),
        )
        return res_f_expanded, mol_f_combined, targets

    def _tensorize_eval(self, idx):
        record = self.batches[idx][0]
        try:
            obj = self._load_pocket(record['key'])
            if obj is None or not hasattr(obj, 'pocket_residues'):
                return None
            obj.pK = float(record['pK'])
        except Exception:
            return None

        n_use = min(self.decoy_count, len(obj.decoys)) if obj.decoys else 0
        decoys_to_use = obj.decoys[:n_use]

        res_f = tensorize_protein_pocket([obj])

        mol_f_binder, mol_int_binder = tensorize_ligand([obj.ligand])
        pl_int_binder = torch.from_numpy(
            obj.pro_lig_interaction.reshape([-1]).astype(np.float32))
        pK_binder    = torch.tensor([obj.pK], dtype=torch.float32)
        flag_binder  = torch.tensor([1.0])
        label_binder = torch.tensor([1.0])

        all_mol_f   = [mol_f_binder]
        all_mol_int = [mol_int_binder]
        all_pl_int  = [pl_int_binder]
        all_pKs     = [pK_binder]
        all_flags   = [flag_binder]
        all_labels  = [label_binder]

        for decoy_mol in decoys_to_use:
            try:
                mol_f_d, mol_int_d = tensorize_ligand([decoy_mol])
                pl_int_d = torch.from_numpy(
                    np.zeros([decoy_mol.atom_count, obj.res_count],
                             dtype=np.float32).reshape([-1]))
                all_mol_f.append(mol_f_d)
                all_mol_int.append(mol_int_d)
                all_pl_int.append(pl_int_d)
                all_pKs.append(torch.tensor([0.0]))
                all_flags.append(torch.tensor([0.0]))
                all_labels.append(torch.tensor([0.0]))
            except Exception:
                continue

        merged = []
        for i in range(len(mol_f_binder)):
            if isinstance(mol_f_binder[i], list):
                fixed_scope = []
                offset = 0
                for mf in all_mol_f:
                    for (s, l) in mf[i]:
                        fixed_scope.append((offset, l))
                        offset += l
                merged.append(fixed_scope)
            else:
                merged.append(torch.cat([mf[i] for mf in all_mol_f], dim=0))
        mol_f_combined = tuple(merged)

        n_mol = len(all_mol_f)
        fresidues, res_map, res_scope, alpha_coords = res_f
        res_scope_expanded = res_scope * n_mol
        res_f_expanded = (fresidues, res_map, res_scope_expanded, alpha_coords)

        targets = (
            torch.cat(all_mol_int),
            torch.cat(all_pl_int),
            torch.cat(all_pKs),
            torch.cat(all_flags),
            torch.cat(all_labels),
        )
        return res_f_expanded, mol_f_combined, targets

    def __getitem__(self, idx):
        try:
            if self.decoy_count is not None:
                return self._tensorize_eval(idx)
            return self._tensorize_train(idx)
        except Exception:
            return None

    def __del__(self):
        if self._txn is not None:
            try:
                self._txn.abort()
            except Exception:
                pass
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass