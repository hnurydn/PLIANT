#!/usr/bin/env python3
"""Writes a NEW LMDB in which the decoy poses have been placed inside the pocket. The OLD LMDB
is left UNTOUCHED.

Input  : lmdb_planet_v2/{split}.lmdb + cepte/yerlesim/son/{split}_{key}_{start}.npz
Output : lmdb_planet_v2_cepte/{split}.lmdb   (KIP=cepte) -> decoy coordinates placed in the pocket
         lmdb_planet_v2_ilk100/{split}.lmdb  (KIP=kirp)  -> the OLD distant coordinates for the same
                                                            decoy subset (a CONTROL arm matched on
                                                            the number of decoys)
In both modes the decoy list is cut down to the placed subset and the n_decoys field is updated.
Usage: SPLIT=core KIP=cepte python -u build_inpocket_lmdb.py
"""
import os, sys, json, pickle, time
import numpy as np
sys.path.insert(0, "/arf/scratch/hnaydin/PLANET_FIXED_20260917/pipeline")
sys.path.insert(0, "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts")
import lmdb
import chemutils_esm2_vector          # class definitions needed to unpickle
from rdkit import RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

P     = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020"
SRC   = os.environ.get("SRC", f"{P}/lmdb_planet_v2")
KOK   = "/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/yerlesim"
OUT_N = {"cepte": "lmdb_planet_v2_cepte", "kirp": "lmdb_planet_v2_ilk100"}
SPLIT = os.environ.get("SPLIT", "core")
KIP   = os.environ.get("KIP", "cepte")
DST   = os.environ.get("DST", f"{P}/{OUT_N[KIP]}")
# [20 Sep] if the full set was placed, use the _tam queue; otherwise fall back to the _maks100 queue
KUY   = os.environ.get("KUYRUK_DOSYA", "")
if not KUY:
    _tam = f"{KOK}/kuyruk_{SPLIT}_tam.txt"
    KUY  = _tam if os.path.exists(_tam) else (
        f"{KOK}/kuyruk_{SPLIT}" + ("" if SPLIT == "core" else "_maks100") + ".txt")
MAP   = int(os.environ.get("MAPSIZE_GB", "60")) * (1 << 30)
assert KIP in OUT_N, KIP

os.makedirs(DST, exist_ok=True)


def dilimler():
    """key -> [(start, end), ...], ordered"""
    d = {}
    for l in open(KUY):
        l = l.strip()
        if not l: continue
        k, a, b = l.split("\t")
        d.setdefault(k, []).append((int(a), int(b)))
    for k in d: d[k].sort()
    return d


def main():
    t0 = time.time()
    print(f"kuyruk dosyasi: {KUY}", flush=True)
    dil = dilimler()
    meta_eski = json.load(open(f"{SRC}/{SPLIT}_meta.json"))
    src = lmdb.open(f"{SRC}/{SPLIT}.lmdb", readonly=True, lock=False, readahead=False, max_readers=64)
    dst = lmdb.open(f"{DST}/{SPLIT}.lmdb", map_size=MAP, writemap=False)

    meta_yeni, rapor = [], {"yok": 0, "atom_uyusmaz": 0, "eksik_dilim": 0, "yazilan": 0, "decoy": 0, "decoysuz": 0}
    txn_w = dst.begin(write=True)
    with src.begin() as txn_r:
        for i, rec in enumerate(meta_eski):
            k = rec["key"]
            raw = txn_r.get(k.encode())
            if raw is None:
                rapor["yok"] += 1; continue
            obj = pickle.loads(bytes(raw))
            parcalar = dil.get(k, [])
            if not parcalar:
                # pocket with no decoys: kept as is (the binder set must stay IDENTICAL to the old LMDB)
                obj.decoys = []
                obj.decoys_count = 0
                txn_w.put(k.encode(), pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
                r = dict(rec); r["n_decoys"] = 0; meta_yeni.append(r)
                rapor["yazilan"] += 1; rapor["decoysuz"] += 1
                continue
            xyz_l, nat_l, eks = [], [], False
            for (a, b) in parcalar:
                f = f"{KOK}/son/{SPLIT}_{k}_{a}.npz"
                if not os.path.exists(f):
                    eks = True; break
                z = np.load(f)
                if z["xyz"].shape[0] != b - a:      # skip this pocket if the alignment is broken
                    eks = True; break
                xyz_l.append(z["xyz"]); nat_l.append(z["nat"])
            if eks or not xyz_l:
                rapor["eksik_dilim"] += 1; continue
            nat = np.concatenate(nat_l)
            n_yer = len(nat)
            decoys = (obj.decoys or [])[:n_yer]
            if len(decoys) != n_yer:
                rapor["eksik_dilim"] += 1; continue
            # atom-count check (evidence that the alignment holds)
            if any(d.mol.GetNumAtoms() != int(n) for d, n in zip(decoys, nat)):
                rapor["atom_uyusmaz"] += 1; continue
            if KIP == "cepte":
                off = 0
                for blok in xyz_l:
                    for j in range(blok.shape[0]):
                        m = decoys[off + j].mol
                        c = m.GetConformer()
                        x = blok[j, :m.GetNumAtoms()]
                        for ai in range(m.GetNumAtoms()):
                            c.SetAtomPosition(ai, Point3D(float(x[ai, 0]), float(x[ai, 1]), float(x[ai, 2])))
                    off += blok.shape[0]
            obj.decoys = decoys
            obj.decoys_count = len(decoys)
            txn_w.put(k.encode(), pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            r = dict(rec); r["n_decoys"] = len(decoys); meta_yeni.append(r)
            rapor["yazilan"] += 1; rapor["decoy"] += len(decoys)
            if (i + 1) % 500 == 0:
                txn_w.commit(); txn_w = dst.begin(write=True)
                print(f"  [{i+1}/{len(meta_eski)}] yazilan={rapor['yazilan']} "
                      f"{(time.time()-t0)/60:.1f} dk", flush=True)
    txn_w.commit(); dst.sync(); dst.close(); src.close()

    with open(f"{DST}/{SPLIT}_meta.json", "w") as w: json.dump(meta_yeni, w)
    ort = rapor["decoy"] / max(1, rapor["yazilan"])
    print(f"BITTI {SPLIT}/{KIP}: yazilan={rapor['yazilan']}/{len(meta_eski)} "
          f"ort_decoy={ort:.1f} | atlanan: eksik_dilim={rapor['eksik_dilim']} "
          f"atom_uyusmaz={rapor['atom_uyusmaz']} kayit_yok={rapor['yok']} decoysuz={rapor['decoysuz']} | "
          f"{(time.time()-t0)/60:.1f} dk", flush=True)


if __name__ == "__main__":
    main()
