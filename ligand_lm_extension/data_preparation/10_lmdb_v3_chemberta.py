# 10_lmdb_v3_chemberta.py
# lmdb_planet_v2/{train,valid,core}.lmdb icindeki her kompleks icin binder VE
# tum decoy ligandlarina ChemBERTa embedding hesaplayip nesnelere dogrudan
# ekler (obj.ligand.chemberta_feature / obj.decoys[i].chemberta_feature),
# sonucu YENI lmdb_planet_v3/{split}.lmdb'ye yazar. Tek gecis: 09 (embedding
# cache) + 10 (pkl'e gomme) adimlarini birlestiriyor -- LMDB okuma all_pickles
# okumaktan ~100x hizli oldugu icin (25GB vs 441GB, tek memory-mapped dosya)
# ayri bir h5-cache fazina gerek yok.
import os
import sys
import time
import numpy as np
import torch
import lmdb
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from transformers import AutoTokenizer, AutoModel

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Dataset_preparation")

SRC_ROOT = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2"
DST_ROOT = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v3"
SPLITS = ["core", "valid", "train"]  # kucukten buyuge, erken geri bildirim icin

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
EMBEDDING_DIM = 768
INNER_BATCH_SIZE = 128
DST_MAP_SIZE = 200 * 1024**3  # 200GB sanal (LMDB sparse, disk onceden dolmaz)
COMMIT_EVERY = 200  # bu kadar kompleks sonra write-txn commit et


def embed_smiles_batch(smiles_list, tokenizer, model, device):
    if not smiles_list:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    embeddings = []
    for i in range(0, len(smiles_list), INNER_BATCH_SIZE):
        batch = smiles_list[i:i + INNER_BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        embeddings.append(out.pooler_output.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def process_record(obj, tokenizer, model, device):
    """obj: chemutils.ComplexPocket. Binder + tum decoy'lara chemberta_feature ekler."""
    mols_refs = []  # (target_obj, is_binder)
    smiles_list = []

    def safe_smiles(mol_obj):
        try:
            return Chem.MolToSmiles(mol_obj)
        except Exception:
            return None

    binder_smi = safe_smiles(obj.ligand.mol)
    if binder_smi:
        mols_refs.append(obj.ligand)
        smiles_list.append(binder_smi)
    else:
        obj.ligand.chemberta_feature = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    for dec in obj.decoys:
        smi = safe_smiles(dec.mol)
        if smi:
            mols_refs.append(dec)
            smiles_list.append(smi)
        else:
            dec.chemberta_feature = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    embeddings = embed_smiles_batch(smiles_list, tokenizer, model, device)
    for ref, emb in zip(mols_refs, embeddings):
        ref.chemberta_feature = emb.astype(np.float32)

    return obj


def process_split(split, tokenizer, model, device, limit=None):
    src_path = os.path.join(SRC_ROOT, f"{split}.lmdb")
    dst_path = os.path.join(DST_ROOT, f"{split}.lmdb")
    os.makedirs(DST_ROOT, exist_ok=True)

    done_keys = set()
    if os.path.exists(dst_path):
        try:
            dst_env_r = lmdb.open(dst_path, readonly=True, lock=False)
            with dst_env_r.begin() as txn:
                done_keys.update(k for k, _ in txn.cursor())
            dst_env_r.close()
        except Exception:
            pass
    print(f"[{split}] daha once tamamlanmis: {len(done_keys)}")

    src_env = lmdb.open(src_path, readonly=True, lock=False)
    dst_env = lmdb.open(dst_path, map_size=DST_MAP_SIZE)

    with src_env.begin() as src_txn:
        total = src_txn.stat()["entries"]
        cursor = src_txn.cursor()

        write_txn = dst_env.begin(write=True)
        n_since_commit = 0
        n_processed = 0
        n_skipped = 0
        t0 = time.time()

        pbar = tqdm(cursor, total=(limit or total), desc=f"{split}")
        for key, val in pbar:
            if limit is not None and (n_processed + n_skipped) >= limit:
                break
            if key in done_keys:
                n_skipped += 1
                continue
            try:
                obj = pickle.loads(val)
                obj = process_record(obj, tokenizer, model, device)
                write_txn.put(key, pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
                n_processed += 1
                n_since_commit += 1
            except Exception as e:
                tqdm.write(f"HATA ({key}): {e}")
                continue

            if n_since_commit >= COMMIT_EVERY:
                write_txn.commit()
                write_txn = dst_env.begin(write=True)
                n_since_commit = 0

        write_txn.commit()

    elapsed = (time.time() - t0) / 60
    print(f"[{split}] tamamlandi. islenen={n_processed}, atlanan(resume)={n_skipped}, sure={elapsed:.1f} dk")
    src_env.close()
    dst_env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Test icin: split basina max kac kompleks islensin")
    parser.add_argument("--splits", type=str, default=None, help="Virgulle ayrilmis split listesi (varsayilan: hepsi)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    splits = args.splits.split(",") if args.splits else SPLITS
    for split in splits:
        process_split(split, tokenizer, model, device, limit=args.limit)

    print("\nTUM SPLITLER TAMAMLANDI.")


if __name__ == "__main__":
    main()
