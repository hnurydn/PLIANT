#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BINDER-ONLY TEMAS AUROC — duzeltilmis checkpointler (25 Eyl 2026)
===================================================================
Soru: modelin tahmin ettigi atom-rezidu temas haritasi, kristal kompleksin GERCEK temas
haritasini ne kadar dogru buluyor? Decoy KULLANMAZ -> yerlesim/protonasyon/gomulme
artefaktlarindan bagimsiz. Makaledeki eski sayilar (baseline 0,543 / ESM_UW 0,789 /
v4 0,913 / ESM_VG 0,944) H1/H2/H3'lu eski modellerdendi; burada duzeltilmisler olculur.

Kosullar (etiket HER ZAMAN kristal pozdan):
  kristal : ligand kristal pozunda
  kay2 / kay4 : ligand rastgele yonde 2 / 4 A OTELENMIS (saf oteleme -> RMSD tam 2 / 4 A),
                3 tekrar. Poza kor modelde skor degismemeli (negatif kontrol); poza duyarli
                modelde dusus = katkinin "geometriyi okuma" kismi.
Metrik: tum ciftler havuzlanmis AUROC (makaledeki gibi) + kompleks basi AUROC ortalamasi.
Veri: casf_dock/<v1|v2>/<pdb>.pkl (CASF-2016 coreset, ComplexPocket; ligand + pro_lig_interaction)
Kullanim: PLANET_ARCH=<etiket> python -u eval_temas_binder.py [--limit N]
"""
import os, sys, json, pickle, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_dp_v2 as E          # MODELS, load_model, score_full (ayni yollar, ayni birlestirme)

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
            E.cep_blosum(obj)
            lab = np.asarray(obj.pro_lig_interaction, dtype=np.float32)
            m0 = Chem.Mol(obj.ligand.mol)
            Chem.rdmolops.AssignAtomChiralTagsFromStructure(m0)   # kiralite kristal pozdan, bir kez
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
