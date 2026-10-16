#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BINDER-ONLY CONTACT AUROC (25 Sep 2026)
=========================================
Question: how well does the predicted atom-residue contact map match the TRUE contact map of the
crystal complex? This test uses NO decoys, so it is independent of placement, protonation and
burial artefacts.

Conditions (the label ALWAYS comes from the crystal pose):
  kristal     : ligand in its crystal pose
  kay2 / kay4 : ligand TRANSLATED by 2 / 4 A in a random direction (pure translation, so the RMSD
                is exactly 2 / 4 A), 3 repeats. A pose-blind model must not change its score here
                (negative control); for a pose-aware model the drop is the part of its contribution
                that comes from reading geometry.
Metric: AUROC over all pairs pooled, plus the mean of the per-complex AUROCs.
Data: casf_dock/<v1|v2>/<pdb>.pkl (CASF-2016 core set, ComplexPocket; ligand + pro_lig_interaction)
Usage: PLANET_ARCH=<label> python -u eval_contact_auroc.py [--limit N]
"""
import os, sys, json, pickle, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_docking_power as E          # MODELS, load_model, score_full (same paths, same batching)

KAY = [("kay2", 2.0), ("kay4", 4.0)]
N_TEKRAR = 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from sklearn.metrics import roc_auc_score
    import importlib
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    dev = torch.device("cpu")
    only = os.environ.get("PLANET_ARCH", "").strip()
    sel = [x for x in E.MODELS if x[0] == only]
    if not sel:
        raise SystemExit("PLANET_ARCH eslesmedi: %r" % only)
    label, mod, moddir, ckpt, cu_name, kind, pose_aware = sel[0]
    out = os.path.join(E.OUTDIR, "temas_binder_%s.json" % label)
    rep = json.load(open(os.path.join(E.POCK, "prep_report.json")))
    ids = sorted(r["pdb"] for r in rep["ok"])
    if a.limit: ids = ids[:a.limit]
    print("=== BINDER-ONLY TEMAS: %s (poza %s) hedef %d ===" % (label, "duyarli" if pose_aware else "kor", len(ids)), flush=True)
    cu = importlib.import_module(cu_name); print("  chemutils:", cu.__file__, flush=True)
    model = E.load_model(mod, moddir, ckpt, dev)
    rng = np.random.default_rng(0)
    pool = {"kristal": ([], [])}; pool.update({k: ([], []) for k, _ in KAY})
    per = {}; hata = 0
    for n, pid in enumerate(ids, 1):
        try:
            obj = pickle.load(open(os.path.join(E.POCK, kind, pid + ".pkl"), "rb"))
            E.pocket_to_blosum(obj)
            lab = np.asarray(obj.pro_lig_interaction, dtype=np.float32)
            m0 = Chem.Mol(obj.ligand.mol)
            Chem.rdmolops.AssignAtomChiralTagsFromStructure(m0)   # chirality is taken from the crystal pose, once
            xyz = m0.GetConformer().GetPositions()
            mols, tags = [m0], ["kristal"]
            for k, d in KAY:
                for _ in range(N_TEKRAR):
                    v = rng.normal(size=3); v = v / np.linalg.norm(v) * d
                    m = Chem.Mol(m0); conf = m.GetConformer()
                    for i, (x, y, z) in enumerate(xyz + v):
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                    mols.append(m); tags.append(k)
            ligs = [cu.Mol(m) for m in mols]
            _, cmaps = E.score_full(model, obj, cu, ligs, dev)
            if cmaps[0].shape != lab.shape:
                raise RuntimeError("boyut: tahmin %s etiket %s" % (cmaps[0].shape, lab.shape))
            y = lab.ravel()
            r = {}
            for t in pool:
                ps = [c.ravel() for c, tg in zip(cmaps, tags) if tg == t]
                for p in ps:
                    pool[t][0].append(y); pool[t][1].append(p)
                if 0 < y.sum() < y.size:
                    r[t] = float(np.mean([roc_auc_score(y, p) for p in ps]))
            per[pid] = r
        except Exception as ex:
            hata += 1; per[pid] = {"error": str(ex)[:160]}
        if n % 50 == 0: print("    %d/%d" % (n, len(ids)), flush=True)
    res = {"label": label, "ckpt": ckpt, "pose_aware": pose_aware, "n_hata": hata, "per_target": per}
    for t, (ys, ps) in pool.items():
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        vals = [v[t] for v in per.values() if t in v]
        res[t] = {"havuz_auroc": round(float(roc_auc_score(yy, pp)), 4),
                  "kompleks_ort_auroc": round(float(np.mean(vals)), 4), "n": len(vals)}
    json.dump(res, open(out, "w"))
    print("\nhata %d / %d" % (hata, len(ids)))
    for t in pool:
        print("  %-8s havuz AUROC %.4f   kompleks ort %.4f  (n=%d)" % (t, res[t]["havuz_auroc"], res[t]["kompleks_ort_auroc"], res[t]["n"]), flush=True)
    print("Sonuc:", out)


if __name__ == "__main__":
    main()
