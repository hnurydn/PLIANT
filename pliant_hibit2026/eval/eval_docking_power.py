#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASF-2016 DOCKING POWER, with random tie-breaking and a contact-agreement channel (25 Sep 2026)
=================================================================================================
Two corrections on top of the earlier eval_casf_docking_power.py (16 Sep):

1) TIES: the earlier summary used np.argsort(-s), which always picks the first pose when the
   scores are equal. In the CASF mol2 files the first pose is within 2 A for 54.6% of the
   targets, so a pose-blind model appeared to "succeed" on 52% of them, which is an artefact.
   Here ties are broken AT RANDOM and averaged over 200 repeats, and the chance expectation is
   reported for every target.

2) AGREEMENT SCORE (the pairwise contact counterpart of Eq. 10 in PLANET v2.0): the mean
   log-likelihood between the predicted contact map p_ij and the pose's GEOMETRIC contact map
   c_ij (atom-residue <= 4.0 A, the same rule as the training label):
       agreement = mean_ij [ c*log p + (1-c)*log(1-p) ]     (higher is better)
   For a pose-blind model p does not depend on the pose, so the score separates poses through c
   alone, which asks whether the model knows the correct binding mode for this pocket.
   PRE-DECLARED: the primary channel is agreement; the affinity and mean-contact channels are
   secondary and are kept for comparison with the earlier measurement.

Negative control: for a pose-blind model the affinity and mean-contact channels must be constant
within every target (tum_berabere_orani ~ 1.0), and with the tie correction top1 should come out
at chance.

Usage: PLANET_ARCH=<label> PLANET_OUT=<file.json> python -u eval_docking_power.py [--limit N]
"""
import os, sys, json, pickle, importlib, argparse, gc
import numpy as np

ROOT = "/arf/scratch/hnaydin/PLANET_ORIGINAL"
BASE = os.path.join(ROOT, "Planet_model")
SCRIPTS = os.path.join(BASE, "Scripts")

# --- 15 Oct 2026: BLOSUM pocket features (the baseline / ABL1 arm) ----------------
# The CASF pocket pickles carry ESM-2 (512-dim) features, but the BLOSUM arm expects 20
# dims. Importing that model module sets chemutils.FEATURE_DIM from 512 to 20, after which
# tensorize silently truncates the mismatched feature vector WITHOUT raising (the first 20
# ESM dimensions are meaningless). Here the features are rebuilt from the residue TYPE;
# Residue.residue_type is available, so no coordinate matching is needed.
_BLOSUM62 = None


def pocket_to_blosum(obj):
    """If PLANET_BLOSUM=1, replace the pocket's residue features with BLOSUM62 rows."""
    global _BLOSUM62
    if os.environ.get("PLANET_BLOSUM", "0") != "1":
        return obj
    if _BLOSUM62 is None:
        import importlib.util
        _sp = importlib.util.spec_from_file_location(
            "orjchem", os.path.join(ROOT, "orig_ctrl", "code", "chemutils.py"))
        _oc = importlib.util.module_from_spec(_sp)
        _sp.loader.exec_module(_oc)
        _BLOSUM62 = _oc.BLOSUM62
        print("  BLOSUM cebi ETKIN (%d rezidu turu)" % len(_BLOSUM62), flush=True)
    for r in obj.pocket_residues:
        t = r.residue_type
        if t not in _BLOSUM62:
            raise SystemExit("BLOSUM62'de olmayan rezidu turu: %r" % t)
        r.feature = np.asarray(_BLOSUM62[t], dtype=np.float32).reshape(-1)
    return obj
# ──────────────────────────────────────────────────────────────────────────────

SUMLN = os.path.join(ROOT, "sumln_matrix")
SUMAFF = os.path.join(SUMLN, "sumaff_dude")
H3 = os.path.join(SUMLN, "h3_dude")
KAPI = os.path.join(ROOT, "kapi")
V4K = os.path.join(ROOT, "v4kont")
CASF = os.path.join(ROOT, "casf2016_official/CASF-2016")
DOCK = os.path.join(CASF, "decoys_docking")
POCK = os.path.join(ROOT, "casf_dock")
OUTDIR = os.path.join(POCK, "dp_v2")
RMSD_CUT = 2.0
CONTACT_CUT = 4.0
N_TIE = 200

# same path order as the earlier script (verified: chemutils is picked up from Scripts/)
for p in (BASE, SCRIPTS, SUMLN):
    if p not in sys.path:
        sys.path.insert(0, p)

# Model registry. Fields: label, model module, module directory (None = sumln_matrix),
# checkpoint, chemutils module, pickle format, pose-aware.
#
# This published copy lists only the three PLIANT variants reported in the HIBIT 2026
# abstract. The copy used on the cluster carried ~40 further entries for other arms of
# the project; they are removed here because they are unrelated to the abstract. The
# evaluation code below is unchanged; see PROVENANCE.json for the original filename
# and md5.
#
# Note on the pose-aware flag: the cluster registry marked ABL2_P0_700k (PLIANT-ESM)
# as pose-aware, which is a labelling error. That variant has no distance-biased vector
# attention and gives every pose of a ligand the same affinity (its tie ratio in this
# test is 1.000). The flag only selects a printed label and a JSON field and does not
# affect the computation; it is set to False here to match the measurement.
MODELS = [
    # PLIANT - BLOSUM62 residue features; run with PLANET_BLOSUM=1
    ("ABL1_P0_700k", "pliant_model",
     os.path.join(ROOT, "anzu_kod_ikinci/ablasyon"),
     os.path.join(SUMAFF, "ckpt/ABL1_P0_700k/PLANET.iter-700000"),
     "chemutils_esm2_vector", "v2", False),
    # PLIANT-ESM - ESM-2 residue features, no pose-aware components
    ("ABL2_P0_700k", "pliant_esm_model",
     os.path.join(ROOT, "anzu_kod_ikinci/ablasyon"),
     os.path.join(SUMAFF, "ckpt/ABL2_P0_700k/PLANET.iter-700000"),
     "chemutils_esm2_vector", "v2", False),
    # PLIANT-ESM-Pose - cross-modal gating + distance-biased vector attention
    ("BYER0_700k", "pliant_esm_pose_model",
     os.path.join(ROOT, "birlesik"),
     os.path.join(SUMAFF, "ckpt/BYER0_700k/PLANET.iter-700000"),
     "chemutils_esm2_vector", "v2", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    device = torch.device("cpu")
    only = os.environ.get("PLANET_ARCH", "").strip()
    sel = [x for x in MODELS if x[0] == only]
    if not sel:
        raise SystemExit("PLANET_ARCH eslesmedi: %r  secenekler: %s" % (only, [x[0] for x in MODELS]))
    label, mod, moddir, ckpt, cu_name, kind, pose_aware = sel[0]
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, os.environ.get("PLANET_OUT", "dp_v2_%s.json" % label))

    rep = json.load(open(os.path.join(POCK, "prep_report.json")))
    ids = sorted(r["pdb"] for r in rep["ok"]
                 if os.path.exists(os.path.join(DOCK, r["pdb"] + "_decoys.mol2"))
                 and os.path.exists(os.path.join(DOCK, r["pdb"] + "_rmsd.dat")))
    if a.limit:
        ids = ids[:a.limit]
    print("=== DOCKING POWER v2: %s (poza %s) ===" % (label, "duyarli" if pose_aware else "kor"))
    print("ckpt:", ckpt, "  hedef:", len(ids), flush=True)
    if not os.path.exists(ckpt):
        raise SystemExit("checkpoint yok: " + ckpt)

    cu = importlib.import_module(cu_name)
    print("  chemutils:", cu.__file__, flush=True)
    model = load_model(mod, moddir, ckpt, device)

    per = {}
    for n, pid in enumerate(ids, 1):
        try:
            mols, dropped = read_poses_template(os.path.join(DOCK, pid + "_decoys.mol2"))
            rm = read_rmsd(os.path.join(DOCK, pid + "_rmsd.dat"))
            keep = [(nm, m, rm[nm]) for nm, m in mols if nm in rm]
            if len(keep) < 5:
                per[pid] = {"error": "poz<5 (atilan %d)" % dropped}; continue
            obj = pickle.load(open(os.path.join(POCK, kind, pid + ".pkl"), "rb"))
            pocket_to_blosum(obj)
            # 25 Sep: mol_batch_to_graph recomputed chirality from each pose's 3D coordinates
            # (it is an atom feature), so a pose-blind model produced 20 different scores
            # across the 94 poses of 1g2k. Chirality is now taken ONCE from the pose closest
            # to native and copied verbatim to every pose.
            from rdkit import Chem
            src = Chem.Mol(keep[int(np.argmin([r for _, _, r in keep]))][1])
            Chem.rdmolops.AssignAtomChiralTagsFromStructure(src)
            tags = [a.GetChiralTag() for a in src.GetAtoms()]
            for _, m, _ in keep:
                for a, t in zip(m.GetAtoms(), tags):
                    a.SetChiralTag(t)
            ligs = [cu.Mol(m) for _, m, _ in keep]
            aff, cmaps = score_full(model, obj, cu, ligs, device)
            rx, ri, nr = residue_atoms(obj)
            agr, dic, bll, cmean, ncon = [], [], [], [], []
            for (_, m, _), p in zip(keep, cmaps):
                c = geom_contact(m, rx, ri, nr)
                if c.shape != p.shape:
                    raise RuntimeError("harita boyu: geom %s model %s" % (c.shape, p.shape))
                agr.append(agreement(p, c)); dic.append(dice(p, c)); bll.append(balanced_ll(p, c))
                cmean.append(float(p.mean())); ncon.append(int(c.sum()))
            per[pid] = {"aff": [float(x) for x in aff], "con": cmean, "dice": dic, "uyum_dengeli": bll,
                        "uyum_ham": agr, "n_temas": ncon, "atilan_poz": dropped,
                        "rmsd": [r for _, _, r in keep]}
        except Exception as ex:
            per[pid] = {"error": str(ex)[:160]}
        if n % 25 == 0:
            print("    %d/%d" % (n, len(ids)), flush=True)

    res = {"label": label, "ckpt": ckpt, "pose_aware": pose_aware,
           "n_hata": sum("error" in v for v in per.values()),
           "atilan_poz_toplam": sum(v.get("atilan_poz", 0) for v in per.values()),
           "dice": summarize(per, "dice"), "uyum_dengeli": summarize(per, "uyum_dengeli"),
           "uyum_ham": summarize(per, "uyum_ham"), "n_temas": summarize(per, "n_temas"),
           "affinity": summarize(per, "aff"), "contact_mean": summarize(per, "con"),
           "per_target": per}
    json.dump(res, open(out, "w"))
    print("\nhata: %d / %d hedef   sablona uymadigi icin atilan poz: %d"
          % (res["n_hata"], len(ids), res["atilan_poz_toplam"]))
    # n_temas: a model-independent reference - the naive "pick the pose with the most contacts" rule
    for ch in ("dice", "uyum_dengeli", "uyum_ham", "affinity", "contact_mean", "n_temas"):
        S = res[ch]
        print("  %-12s top1 %6s (sans %s)  top2 %6s  top3 %6s  rho %6s  berabere %s  n=%s"
              % (ch, S["top1"], S["sans_top1"], S["top2"], S["top3"], S["rho_vs_rmsd"],
                 S["tum_berabere_orani"], S["n_targets"]), flush=True)
    errs = [v["error"] for v in per.values() if "error" in v][:5]
    if errs:
        print("  ornek hatalar:", errs)
    print("Sonuc:", out)


if __name__ == "__main__":
    main()
