#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RESMI CASF-2016 DOCKING POWER — DUZELTILMIS CHECKPOINTLER + UYUM SKORU (25 Eyl 2026)
=====================================================================================
eval_casf_docking_power.py'nin (16 Eyl) uzerine iki duzeltme:

1) BERABERLIK: eski ozet np.argsort(-s) ile esit skorlarda hep ilk pozu seciyordu;
   CASF mol2 dosyalarinda ilk poz hedeflerin %54,6'sinda <=2 A oldugu icin poza kor
   modeller %52 "basari" aliyordu (artefakt). Burada beraberlik RASTGELE bozulur
   (200 tekrar ortalamasi) ve her hedef icin sans beklentisi de raporlanir.

2) UYUM SKORU (PLANET v2.0 Denklem 10'un ikili temas karsiligi): modelin tahmin ettigi
   temas haritasi p_ij ile pozun GEOMETRIK temas haritasi c_ij (atom-rezidu <=4.0 A,
   egitim etiketiyle ayni kural) arasindaki ortalama log-olabilirlik:
       uyum = mean_ij [ c*log p + (1-c)*log(1-p) ]     (buyuk = iyi)
   Poza kor modellerde p pozdan bagimsizdir, skor yalniz c uzerinden pozlari ayirir ->
   "model bu cep icin dogru baglanma modunu biliyor mu" sorusu. ONCEDEN ILAN: birincil
   kanal = uyum; aff ve temas-ortalamasi eski olcumle kiyas icin ikincil.

Negatif kontrol: poza kor modellerde aff ve temas-ortalamasi her hedefte sabit olmali
(tum_berabere_orani ~1.0); beraberlik duzeltmesiyle top1 ~ sans cikmali.

Kullanim: PLANET_ARCH=<etiket> PLANET_OUT=<dosya.json> python -u eval_docking_power.py [--limit N]
"""
import os, sys, json, pickle, importlib, argparse, gc
import numpy as np

ROOT = "/arf/scratch/hnaydin/PLANET_ORIGINAL"
BASE = os.path.join(ROOT, "Planet_model")
SCRIPTS = os.path.join(BASE, "Scripts")

# ─── 15 Eki 2026: BLOSUM cebi (Baseline/ABL1 kolu) ────────────────────────────
# CASF cep pickle'lari ESM-2 (512) oznitelikli. BLOSUM kolu 20 boyut bekler; model modulu
# import edilince chemutils.FEATURE_DIM 512->20 oluyor ve tensorize UYUMSUZ ozniteligi
# HATA VERMEDEN kirpar (ilk 20 ESM boyutu = anlamsiz). Burada oznitelik rezidu TURUNDEN
# yeniden kurulur. Residue.residue_type mevcut -> koordinat eslemesi gerekmiyor.
_BLOSUM62 = None


def pocket_to_blosum(obj):
    """PLANET_BLOSUM=1 ise cebin rezidu ozniteliklerini BLOSUM62 satirlariyla degistirir."""
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

# eski script ile ayni yol sirasi (dogrulanmis: chemutils Scripts'ten gelir)
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
            # 25 Eyl: mol_batch_to_graph kiraliteyi her pozun 3B'sinden yeniden hesapliyordu
            # (atom ozelligi!) -> poza kor model 1g2k'da 94 poza 20 farkli skor verdi.
            # Kiralite native'e en yakin pozdan BIR KEZ alinir, tum pozlara aynen yazilir.
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
    # n_temas: modelden bagimsiz referans — "en cok temas eden pozu sec" basit kurali
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
