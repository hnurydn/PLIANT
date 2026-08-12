"""
PLANET_datautils_crossdocked.py
================================
DataLoader for CrossDocked 2020 fine-tuning.

Differences from the PDBbind DataLoader:
  - each batch = 12 random binders from the same pocket (all complex_label=1)
  - no decoys
  - pK_flag=1 (Vina-based pK values)
  - each binder is its own LMDB record -- grouped by pocket_name in the meta JSON

LMDB meta JSON structure:
  [
    {"key": "1433B__2bq0_5f74_amp", "pocket_name": "1433B_HUMAN_1_240_pep_0",
     "pdb_id": "2bq0", "pK": 5.12},
    ...
  ]
"""

import os, pickle, random, lmdb, json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from chemutils_esm2_vector import (
    tensorize_protein_pocket,
    tensorize_ligand,
)


class CrossDockedDataset(Dataset):

    def __init__(self, lmdb_path, batch_size=12, shuffle=True):
        self.lmdb_path  = lmdb_path
        self.batch_size = batch_size

        self._env = None
        self._txn = None
        self._pid = None

        # Meta JSON
        meta_path = lmdb_path.rstrip('/\\')
        if meta_path.endswith('.lmdb'):
            meta_path = meta_path[:-5]
        meta_path += '_meta.json'

        # Receptor bazlı meta varsa onu kullan (boyut uyumsuzluğunu önler)
        receptor_meta = meta_path.replace("_meta.json", "_meta_receptor.json")
        filtered_meta = meta_path.replace("_meta.json", "_meta_filtered.json")
        if os.path.exists(receptor_meta):
            meta_path = receptor_meta
            print(f"✅ Receptor bazlı meta kullanılıyor: {meta_path}")
        elif os.path.exists(filtered_meta):
            meta_path = filtered_meta
            print(f"✅ Filtrelenmiş meta kullanılıyor: {meta_path}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')

        with open(meta_path, 'r') as f:
            all_records = json.load(f)

        # Receptor bazlı gruplama (aynı receptor = aynı protein boyutu)
        self.pocket_groups = defaultdict(list)
        for r in all_records:
            group_key = r.get('receptor_id', r.get('pdb_id', r['pocket_name']))
            self.pocket_groups[group_key].append(r)

        self.pocket_names = list(self.pocket_groups.keys())
        self.batches = self._build_batches(shuffle)

        total_binders = len(all_records)
        print(f'✅ CrossDocked Dataset hazır: {len(self.batches)} batch '
              f'({len(self.pocket_names)} receptor, '
              f'{total_binders} binder, '
              f'ort. {total_binders/len(self.pocket_names):.1f} binder/receptor, '
              f'batch_size={batch_size})')

    def _build_batches(self, shuffle=True):
        pocket_names = self.pocket_names.copy()
        if shuffle:
            random.shuffle(pocket_names)
        batches = []
        for pocket_name in pocket_names:
            records = self.pocket_groups[pocket_name].copy()
            random.shuffle(records)
            batch = records[:self.batch_size]
            if len(batch) > 0:
                batches.append(batch)
        return batches

    def reshuffle(self):
        self.batches = self._build_batches(shuffle=True)

    def __len__(self):
        return len(self.batches)

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            if self._env is not None:
                try:
                    self._txn.abort()
                    self._env.close()
                except:
                    pass
            self._env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True,
                lock=False, readahead=True, meminit=False, max_readers=256,
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

    def __getitem__(self, idx):
        try:
            return self._tensorize(idx)
        except Exception:
            return None

    def _tensorize(self, idx):
        batch_records = self.batches[idx]

        pockets = []
        for r in batch_records:
            obj = self._load_pocket(r['key'])
            if obj is None or not hasattr(obj, 'pocket_residues'):
                continue
            obj.pK = float(r.get('pK', 0.0))
            pockets.append(obj)

        if not pockets:
            return None

        all_mol_f   = []
        all_mol_int = []
        all_pl_int  = []
        all_pKs     = []
        all_flags   = []
        all_labels  = []
        valid_pockets = []

        for pocket in pockets:
            try:
                mol_f, mol_int = tensorize_ligand([pocket.ligand])
                pl_int = torch.from_numpy(
                    pocket.pro_lig_interaction.reshape([-1]).astype(np.float32))
                all_mol_f.append(mol_f)
                all_mol_int.append(mol_int)
                all_pl_int.append(pl_int)
                all_pKs.append(torch.tensor([pocket.pK], dtype=torch.float32))
                all_flags.append(torch.tensor([1.0]))
                all_labels.append(torch.tensor([1.0]))
                valid_pockets.append(pocket)
            except Exception:
                continue

        if not valid_pockets:
            return None

        # Her binder için ayrı protein feature — res_count farklı olabilir
        all_res_f = [tensorize_protein_pocket([p]) for p in valid_pockets]

        # Mol feature'larını birleştir
        mol_f_ref = all_mol_f[0]
        merged = []
        for i in range(len(mol_f_ref)):
            if isinstance(mol_f_ref[i], list):
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

        # Protein feature'larını birleştir — her binder kendi res_scope'u ile
        n_mol = len(valid_pockets)
        all_fresidues    = torch.cat([r[0] for r in all_res_f], dim=0)
        all_alpha_coords = torch.cat([r[3] for r in all_res_f], dim=0)

        # res_map: her binder kendi bloğunda
        total_res = sum(r[0].shape[0] for r in all_res_f)
        combined_res_map = torch.zeros(total_res, total_res, 1)
        offset = 0
        for r in all_res_f:
            n = r[0].shape[0]
            combined_res_map[offset:offset+n, offset:offset+n] = r[1][:n, :n]
            offset += n

        # res_scope: offset'leri kümülatif güncelle
        combined_res_scope = []
        res_offset = 0
        for r in all_res_f:
            for (s, l) in r[2]:
                combined_res_scope.append((res_offset + s, l))
            res_offset += r[0].shape[0]

        res_f_expanded = (all_fresidues, combined_res_map, combined_res_scope, all_alpha_coords)

        # Targets — 6 elemanlı (contrastive için group_id dahil)
        targets = (
            torch.cat(all_mol_int),
            torch.cat(all_pl_int),
            torch.cat(all_pKs),
            torch.cat(all_flags),
            torch.cat(all_labels),
            # 5 elemanlı — ESM_VG compute_loss beklentisiyle uyumlu
        )

        return res_f_expanded, mol_f_combined, targets

    def __del__(self):
        if self._txn is not None:
            try: self._txn.abort()
            except: pass
        if self._env is not None:
            try: self._env.close()
            except: pass