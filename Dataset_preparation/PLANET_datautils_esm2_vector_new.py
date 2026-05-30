import os, pickle, random, lmdb, json
import torch
from torch.utils.data import Dataset


def _merge_single_outputs(outputs_list):
    """
    Her biri tensorize_all([single_obj]) çıktısı olan listeyi
    tek bir batch çıktısına birleştirir.

    outputs_list[i] = (res_batch_tuple, mol_batch_tuple, mol_int, pl_int, pK, pK_flag, label)
    """
    import torch

    # ── res_feature_batch birleştir ──────────────────────────────────────────
    all_fresidues, all_res_scopes, all_alpha = [], [], []
    total_res = 0
    for out in outputs_list:
        fresidues, res_map, res_scope, alpha = out[0]
        n = fresidues.shape[0]
        all_fresidues.append(fresidues)
        all_alpha.append(alpha)
        all_res_scopes.append((total_res, n))
        total_res += n

    # res_map: blok-diyagonal birleştirme
    combined_res_map = torch.zeros(total_res, total_res, 1)
    ptr = 0
    for out in outputs_list:
        _, res_map, _, _ = out[0]
        n = res_map.shape[0]
        combined_res_map[ptr:ptr+n, ptr:ptr+n, :] = res_map
        ptr += n

    res_batch = (
        torch.cat(all_fresidues, dim=0),
        combined_res_map,
        all_res_scopes,
        torch.cat(all_alpha, dim=0),
    )

    # ── mol_feature_batch birleştir ──────────────────────────────────────────
    all_fatoms, all_fbonds, all_agraphs, all_bgraphs, all_lig_scopes, all_lig_coords = [], [], [], [], [], []
    total_atoms = 0
    total_bonds = 1  # padding bond

    for out in outputs_list:
        fatoms, fbonds, agraph, bgraph, lig_scope, lig_coords = out[1]
        n_atoms = fatoms.shape[0]
        n_bonds = fbonds.shape[0]   # padding dahil

        all_fatoms.append(fatoms)
        all_lig_coords.append(lig_coords)
        all_lig_scopes.append((total_atoms, n_atoms))

        # agraph: atom → bond indeksi, offset = total_bonds - 1 (padding çıkarılır)
        agraph_off = agraph.clone()
        mask_a = agraph_off != 0
        agraph_off[mask_a] += (total_bonds - 1)
        all_agraphs.append(agraph_off)

        # bgraph: bond → bond indeksi, aynı offset
        bgraph_off = bgraph.clone()
        mask_b = bgraph_off != 0
        bgraph_off[mask_b] += (total_bonds - 1)
        all_bgraphs.append(bgraph_off[1:])

        # fbonds: padding bond'u (index 0) atla, geri kalanı ekle
        all_fbonds.append(fbonds[1:])

        total_atoms += n_atoms
        total_bonds += (n_bonds - 1)   # padding hariç gerçek bond sayısı

    # Başa tek padding satırı ekle
    bond_dim = all_fbonds[0].shape[1] if all_fbonds else 37
    padding  = torch.zeros(1, bond_dim)
    padding_bg = torch.zeros(1, 10, dtype=torch.long)

    mol_batch = (
        torch.cat(all_fatoms,     dim=0),
        torch.cat([padding] + all_fbonds, dim=0),
        torch.cat(all_agraphs,    dim=0),
        torch.cat([padding_bg] + all_bgraphs, dim=0),
        all_lig_scopes,
        torch.cat(all_lig_coords, dim=0),
    )

    # ── targets birleştir ────────────────────────────────────────────────────
    mol_ints = torch.cat([out[2] for out in outputs_list])
    pl_ints  = torch.cat([out[3] for out in outputs_list])
    pKs      = torch.cat([out[4] for out in outputs_list])
    pK_flags = torch.cat([out[5] for out in outputs_list])
    labels   = torch.cat([out[6] for out in outputs_list])

    return res_batch, mol_batch, (mol_ints, pl_ints, pKs, pK_flags, labels)


class ProLigDataset(Dataset):
    def __init__(self, lmdb_path, batch_size, shuffle=True, decoy_flag=True,
                 binder_per_batch=4, decoy_count=None):
        """
        Tensor LMDB versiyonu: her kayıt zaten tensorize edilmiş.
        _tensorize() sadece LMDB'den okur, tensorize_all() çağırmaz.
        """
        self.lmdb_path        = lmdb_path
        self.batch_size       = batch_size
        self.decoy_flag       = decoy_flag
        self.binder_per_batch = binder_per_batch
        self.decoy_count      = decoy_count
        self._env             = None

        # ── Meta JSON yükle ──────────────────────────────────────────────────
        meta_path = lmdb_path.rstrip('/').rstrip('\\')
        if meta_path.endswith('.lmdb'):
            meta_path = meta_path[:-5]
        meta_path += '_meta.json'

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Meta JSON bulunamadı: {meta_path}')

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.binders = [m for m in meta if not m['is_decoy']]
        self.decoys  = [m for m in meta if m['is_decoy']]

        # ── Batch listesi ────────────────────────────────────────────────────
        if decoy_count is not None:
            self.batches = self._build_eval_batches()
            mode = f'eval (her binder için max {decoy_count} decoy)'
        else:
            self.batches = self._build_train_batches(shuffle)
            mode = 'train'

        print(f'✅ LMDB Tensor Dataset hazır [{mode}]: {len(self.batches)} batch '
              f'({len(self.binders)} binder, {len(self.decoys)} decoy)')

    def _build_train_batches(self, shuffle=True):
        binders = self.binders.copy()
        if shuffle:
            random.shuffle(binders)

        from itertools import cycle
        decoy_cycle = cycle(self.decoys) if self.decoys else iter([])
        n_batches   = len(binders) // self.binder_per_batch

        batches = []
        for i in range(n_batches):
            b     = binders[i * self.binder_per_batch:(i + 1) * self.binder_per_batch]
            d     = [next(decoy_cycle) for _ in range(self.batch_size - self.binder_per_batch)] if self.decoys else []
            batch = b + d
            random.shuffle(batch)
            batches.append(batch)
        return batches

    def _build_eval_batches(self):
        decoy_map = {}
        for d in self.decoys:
            decoy_map.setdefault(d['pdb_id'], []).append(d)

        batches = []
        for binder in self.binders:
            decoys = decoy_map.get(binder['pdb_id'], [])[:self.decoy_count]
            batches.append([binder] + decoys)
        return batches

    def reshuffle(self):
        if self.decoy_count is None:
            self.batches = self._build_train_batches(shuffle=True)

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True, lock=False,
                readahead=True, meminit=False,
            )
        return self._env

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        try:
            return self._load_batch(idx)
        except Exception:
            return None

    def _load_batch(self, idx):
        env = self._get_env()
        outputs_list = []

        with env.begin() as txn:
            for record in self.batches[idx]:
                raw = txn.get(record['key'].encode())
                if raw is None:
                    continue
                # Hazır tensor tuple'ı deserialize et
                outputs = list(pickle.loads(raw))
                import torch
                is_decoy = record['is_decoy']
                if is_decoy:
                    outputs[6] = torch.zeros(1)
                    outputs[5] = torch.zeros(1)
                else:
                    outputs[6] = torch.ones(1)
                    outputs[5] = torch.ones(1) if record['pK'] > 0 else torch.zeros(1)
                outputs_list.append(tuple(outputs))

        if not outputs_list:
            return None

        # Tek sample ise direkt döndür, çok sample ise birleştir
        if len(outputs_list) == 1:
            out = outputs_list[0]
            res_f   = out[0]
            mol_f   = out[1]
            targets = (out[2], out[3], out[4], out[5], out[6])
            return res_f, mol_f, targets

        # Batch birleştirme
        return _merge_single_outputs(outputs_list)

    def __del__(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
