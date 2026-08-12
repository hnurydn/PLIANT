import os, pickle, random, lmdb, json
import numpy as np
import torch
from torch.utils.data import Dataset
from chemutils_esm2_vector_ligandlm import tensorize_protein_pocket, tensorize_ligand


class ProLigDataset(Dataset):

    def __init__(self, lmdb_path, batch_size, shuffle=True,
                 decoy_flag=True, binder_per_batch=2, decoy_count=None):

        self.lmdb_path        = lmdb_path
        self.batch_size       = batch_size
        self.decoy_flag       = decoy_flag
        self.binder_per_batch = binder_per_batch
        self.decoy_count      = decoy_count

        # Fork-safe
        self._env = None
        self._txn = None
        self._pid = None

        # ── Meta JSON ──────────────────────────────────────────
        meta_path = lmdb_path.rstrip('/\\')
        if meta_path.endswith('.lmdb'):
            meta_path = meta_path[:-5]
        meta_path += '_meta.json'

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # v2'de sadece binder'lar var
        self.binders = meta

        # ── Batch listesi ──────────────────────────────────────
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

    # ── Batch oluşturucular ────────────────────────────────────

    def _build_train_batches(self, shuffle=True):
        # Her batch = binder_per_batch kadar binder (her biri kendi decoylarıyla)
        binders = self.binders.copy()
        if shuffle:
            random.shuffle(binders)
        n_batches = len(binders) // self.binder_per_batch
        batches = []
        for i in range(n_batches):
            batch = binders[i * self.binder_per_batch:(i + 1) * self.binder_per_batch]
            batches.append(batch)
        return batches

    def _build_eval_batches(self):
        # Eval'da her binder ayrı batch
        return [[b] for b in self.binders]

    def reshuffle(self):
        """Her epoch başında train batch'lerini yeniden karıştır."""
        if self.decoy_count is None:
            self.batches = self._build_train_batches(shuffle=True)

    # ── LMDB erişimi (Fork-safe) ───────────────────────────────

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

    # ── Dataset interface ──────────────────────────────────────

    def __len__(self):
        return len(self.batches)

    def _tensorize_train(self, idx):
        all_res_f    = []
        all_mol_f    = []
        all_mol_int  = []
        all_pl_int   = []
        all_pKs      = []
        all_flags    = []
        all_labels   = []
        all_group_ids = []

        valid_group_idx = 0

        for record in self.batches[idx]:
            try:
                obj = self._load_pocket(record['key'])
                if obj is None or not hasattr(obj, 'pocket_residues'):
                    continue
                obj.pK = float(record['pK'])
            except Exception:
                continue

            # ── Protein özelliklerini tensorize et ──────────────
            res_f = tensorize_protein_pocket([obj])

            # ── Binder ──────────────────────────────────────────
            mol_f_binder, mol_int_binder = tensorize_ligand([obj.ligand])
            pl_int_binder = torch.from_numpy(
                obj.pro_lig_interaction.reshape([-1]).astype(np.float32))

            group_mol_f   = [mol_f_binder]
            group_mol_int = [mol_int_binder]
            group_pl_int  = [pl_int_binder]
            group_pKs     = [torch.tensor([obj.pK], dtype=torch.float32)]
            group_flags   = [torch.tensor([1.0])]
            group_labels  = [torch.tensor([1.0])]
            group_ids     = [torch.tensor([valid_group_idx], dtype=torch.long)]

            # ── Decoylar (aynı pocket'tan, max 11) ──────────────
            n_decoys_available = len(obj.decoys) if obj.decoys else 0
            n_decoys_to_use = min(11, n_decoys_available)

            if n_decoys_available < 11:  # 11'den az decoy'u olanları atla
                continue

            if n_decoys_to_use > 0:
                selected_indices = np.random.choice(
                    n_decoys_available, size=n_decoys_to_use,
                    replace=(n_decoys_available < n_decoys_to_use)
                )
                for idx_d in selected_indices:
                    try:
                        decoy_mol = obj.decoys[idx_d]
                        mol_f_d, mol_int_d = tensorize_ligand([decoy_mol])
                        pl_int_d = torch.from_numpy(
                            np.zeros([decoy_mol.atom_count, obj.res_count],
                                     dtype=np.float32).reshape([-1]))
                        group_mol_f.append(mol_f_d)
                        group_mol_int.append(mol_int_d)
                        group_pl_int.append(pl_int_d)
                        group_pKs.append(torch.tensor([0.0]))
                        group_flags.append(torch.tensor([0.0]))
                        group_labels.append(torch.tensor([0.0]))
                        group_ids.append(torch.tensor([valid_group_idx], dtype=torch.long))
                    except Exception:
                        continue

            # ── res_scope: aynı protein n_mol kez tekrar ────────
            n_mol = len(group_mol_f)
            fresidues, res_map, res_scope, alpha_coords = res_f
            res_scope_expanded = res_scope * n_mol
            res_f_expanded = (fresidues, res_map, res_scope_expanded, alpha_coords)

            all_res_f.append(res_f_expanded)
            all_mol_f.append(group_mol_f)
            all_mol_int.extend(group_mol_int)
            all_pl_int.extend(group_pl_int)
            all_pKs.extend(group_pKs)
            all_flags.extend(group_flags)
            all_labels.extend(group_labels)
            all_group_ids.extend(group_ids)

            valid_group_idx += 1

        if not all_res_f:
            return None

        # ── Tüm pocket'ların res_f'lerini birleştir ─────────────
        all_fresidues    = torch.cat([r[0] for r in all_res_f], dim=0)
        all_alpha_coords = torch.cat([r[3] for r in all_res_f], dim=0)

        # res_map: her pocket kendi bloğunda, cross-pocket sıfır
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

        res_f_combined = (all_fresidues, combined_res_map, combined_res_scope, all_alpha_coords)

        # ── Tüm moleküllerin mol_f'lerini birleştir ──────────────
        flat_mol_f = [mf for group in all_mol_f for mf in group]
        ref = flat_mol_f[0]
        merged = []
        for i in range(len(ref)):
            if isinstance(ref[i], list):
                fixed_scope = []
                atom_offset = 0
                for mf in flat_mol_f:
                    for (s, l) in mf[i]:
                        fixed_scope.append((atom_offset, l))
                        atom_offset += l
                merged.append(fixed_scope)
            else:
                merged.append(torch.cat([mf[i] for mf in flat_mol_f], dim=0))
        mol_f_combined = tuple(merged)

        targets = (
            torch.cat(all_mol_int),
            torch.cat(all_pl_int),
            torch.cat(all_pKs),
            torch.cat(all_flags),
            torch.cat(all_labels),
            torch.cat(all_group_ids),   # ← YENİ: group_id
        )
        return res_f_combined, mol_f_combined, targets

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

        # mol_f tuple elemanlarini birlestir
        # lig_scope (index 4) liste — offset duzeltmesi gerekli
        # diger elemanlar tensor — torch.cat ile birlestir
        merged = []
        for i in range(len(mol_f_binder)):
            if isinstance(mol_f_binder[i], list):
                # lig_scope: her molekulun offset'ini kumulatif hesapla
                fixed_scope = []
                offset = 0
                for mf in all_mol_f:
                    sc = mf[i]  # [(start, length)] formatinda
                    for (s, l) in sc:
                        fixed_scope.append((offset, l))
                        offset += l
                merged.append(fixed_scope)
            else:
                merged.append(torch.cat([mf[i] for mf in all_mol_f], dim=0))
        mol_f_combined = tuple(merged)

        # res_scope ayni protein icin n_mol kez tekrarla
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
