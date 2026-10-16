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

Kullanim: PLANET_ARCH=<etiket> PLANET_OUT=<dosya.json> python -u eval_dp_v2.py [--limit N]
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


def cep_blosum(obj):
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

# etiket, model modulu, modulun dizini (None = sumln_matrix), checkpoint, chemutils, pkl turu, poza duyarli
MODELS = [
    ("ORT180k",       "PLANET_model_gatenorm",        None,   os.path.join(SUMAFF, "ckpt/ORT180k/PLANET.iter-180000"),        "chemutils",             "v1", False),
    ("TOP180k",       "PLANET_model_gatenorm_sumaff", SUMAFF, os.path.join(SUMAFF, "ckpt/TOP180k/PLANET.iter-180000"),        "chemutils",             "v1", False),
    ("ESMUW_FIX140k", "PLANET_model_esm2_sumln_gn",   None,   os.path.join(ROOT, "ESMUW_GN_FIXED_seed42/PLANET.iter-140000"), "chemutils_esm2",        "v2", False),
    ("ESMVG_FIX140k", "PLANET_model_esm2_vector_gating_normalized_sumln_gn", None,
                                                              os.path.join(ROOT, "ESMVG_GN_FIXED_seed42/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True),
    ("V4P_FIX300k",   "PLANET_model_esm2_vector_gating_contrastive_loss_v4_normalized_sumln_gn", None,
                                                              os.path.join(H3, "ckpt/V4PFIX300k/PLANET.iter-300000"),        "chemutils_esm2_vector", "v2", True),
    ("LNCTRL300k",    "PLANET_model_esm2_vector_gating_contrastive_loss_v4_normalized_sumln_gn", None,
                                                              os.path.join(H3, "ckpt/LNCTRL300k/PLANET.iter-300000"),        "chemutils_esm2_vector", "v2", True),
    ("KAPI_KAPI",     "PLANET_model_esm2_vector_gating_normalized_sumln_gn", KAPI,
                                                              os.path.join(SUMAFF, "ckpt/KAPIKAPI/PLANET.iter-15003"),       "chemutils_esm2_vector", "v2", True),
    ("KAPI_TOPLA",    "PLANET_model_esm2_vgadd",      KAPI,   os.path.join(SUMAFF, "ckpt/KAPITOPLA/PLANET.iter-15003"),      "chemutils_esm2_vector", "v2", True),
    ("KAPI_TOPLADROP","PLANET_model_esm2_vgadd_drop", KAPI,   os.path.join(SUMAFF, "ckpt/KAPITOPLADROP/PLANET.iter-15003"),  "chemutils_esm2_vector", "v2", True),
    ("V4K_A50k",      "PLANET_model_esm2_vector_gating_contrastive_loss_v4_normalized_sumln_gn", V4K,
                                                              os.path.join(SUMAFF, "ckpt/V4KA/PLANET.iter-50000"),           "chemutils_esm2_vector", "v2", True),
    ("V4K_B50k",      "PLANET_model_v4s",             V4K,    os.path.join(SUMAFF, "ckpt/V4KB/PLANET.iter-50000"),           "chemutils_esm2_vector", "v2", True),
    ("ESMVG_ESKI1.9M","PLANET_model_esm2_vector_gating_normalized_sumln_gn", None,
                                                              os.path.join(SUMLN, "ESMVGNORM_SUMLN_GN_seed42/PLANET.iter-1900000"), "chemutils_esm2_vector", "v2", True),
]

# 25 Eyl: tezdeki GN matrisi (eski nesil, H1/H2/H3'lu egitim) — gorev 5 dogrulamasi icin (indeks 12-23)
for _lbl, _mod, _tpl, _cu, _kind, _pa in [
        ("ESKI_BASE", "PLANET_model_gatenorm", "baseline_GATENORM_seed%d", "chemutils", "v1", False),
        ("ESKI_ESMUW", "PLANET_model_esm2_sumln_gn", "ESMUW_SUMLN_GN_seed%d", "chemutils_esm2", "v2", False),
        ("ESKI_ESMVG", "PLANET_model_esm2_vector_gating_normalized_sumln_gn", "ESMVGNORM_SUMLN_GN_seed%d", "chemutils_esm2_vector", "v2", True),
        ("ESKI_V4P", "PLANET_model_esm2_vector_gating_contrastive_loss_v4_normalized_sumln_gn", "V4PNORM_SUMLN_GN_seed%d", "chemutils_esm2_vector", "v2", True)]:
    for _sd in (42, 123, 456):
        MODELS.append(("%s_s%d" % (_lbl, _sd), _mod, None, os.path.join(SUMLN, _tpl % _sd, "PLANET.iter-1900000"), _cu, _kind, _pa))

# 25 Eyl: VGTOP (ESM_VG + toplam/144, Anzu GPU 2) ve SABITZ (kapi sabit z=0,5, TRUBA 6406032) — indeks 24-25
MODELS.append(("VGTOP140k", "PLANET_model_esm2_vg_top", os.path.join(ROOT, "vgtop"),
               os.path.join(SUMAFF, "ckpt/VGTOP/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("SABITZ140k", "PLANET_model_esm2_vg_sabitz", os.path.join(ROOT, "sabitz"),
               os.path.join(ROOT, "sabitz/SABITZ_seed42/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))

# 25 Eyl: v4 kontrastif 2. TUR (23-24 Eyl), parti 4 cep x (1 binder + 2 decoy), decoy'lar CEPTE
# (lmdb_planet_v2_cepte_prot + zor secim K=60), 180k adim. Indeks 26-27.
MODELS.append(("V4K2_A180k", "PLANET_model_esm2_vector_gating_contrastive_loss_v4_normalized_sumln_gn", V4K,
               os.path.join(SUMAFF, "ckpt/V4K2A/PLANET.iter-180000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("V4K2_B180k", "PLANET_model_v4s", V4K,
               os.path.join(SUMAFF, "ckpt/V4K2B/PLANET.iter-180000"), "chemutils_esm2_vector", "v2", True))

# 28 Eyl: ORJ12 parti duzeni — VGORJ (ESM_VG + toplam/144, 6409377) ve BLORJ (ESM'siz, 6409229), 11.669 ve 60.012 adim. Indeks 28-31.
MODELS.append(("VGORJ11k", "PLANET_model_esm2_vg_top", os.path.join(ROOT, "vgtop"),
               os.path.join(SUMAFF, "ckpt/VGORJ11k/PLANET.iter-11669"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("VGORJ60k", "PLANET_model_esm2_vg_top", os.path.join(ROOT, "vgtop"),
               os.path.join(SUMAFF, "ckpt/VGORJ60k/PLANET.iter-60012"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("BLORJ11k", "PLANET_model_gatenorm_sumaff", SUMAFF,
               os.path.join(SUMAFF, "ckpt/BLORJ11k/PLANET.iter-11669"), "chemutils", "v1", False))
MODELS.append(("BLORJ60k", "PLANET_model_gatenorm_sumaff", SUMAFF,
               os.path.join(SUMAFF, "ckpt/BLORJ60k/PLANET.iter-60012"), "chemutils", "v1", False))
# 29 Eyl: B×C cepte+maske hucreleri (zor K=60, lmdb_planet_v2_cepte_prot, Anzu GPU 6). CM1X12 = 1x(1+11) 140k; CMORJ = ORJ12 11.669 / 60.012. Indeks 32-34.
MODELS.append(("CM1X12_140k", "PLANET_model_esm2_vg_ceptemask", os.path.join(ROOT, "ceptemask"),
               os.path.join(SUMAFF, "ckpt/CM1X12_140k/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("CMORJ11k", "PLANET_model_esm2_vg_ceptemask", os.path.join(ROOT, "ceptemask"),
               os.path.join(SUMAFF, "ckpt/CMORJ11k/PLANET.iter-11669"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("CMORJ60k", "PLANET_model_esm2_vg_ceptemask", os.path.join(ROOT, "ceptemask"),
               os.path.join(SUMAFF, "ckpt/CMORJ60k/PLANET.iter-60012"), "chemutils_esm2_vector", "v2", True))
# 29 Eyl: 25 Eyl kollari (140k, 1x(1+11)). ORJKOL = sig baslik + bolmesiz toplam + clip 200 (uzak decoy);
# CEPTEMASK = cepte_prot (ZORSUZ) + temas maskesi + mol marji (binder >= decoy + 1 pK); MARJ = VGTOP + poz marji (uzak decoy). Indeks 35-37.
MODELS.append(("ORJKOL140k", "PLANET_model_esm2_vg_orj", os.path.join(ROOT, "orjkol"),
               os.path.join(SUMAFF, "ckpt/ORJKOL/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("CEPTEMASK140k", "PLANET_model_esm2_vg_ceptemask", os.path.join(ROOT, "ceptemask"),
               os.path.join(SUMAFF, "ckpt/CEPTEMASK/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("MARJ140k", "PLANET_model_esm2_vg_top", os.path.join(ROOT, "vgtop"),
               os.path.join(SUMAFF, "ckpt/MARJ/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
# 30 Eyl: BIRLESIK tarif, BRL4X3 (4 cep x (1+2)) Anzu GPU6 -- ara checkpoint'ler (egitim 140k'da biter). Indeks 38-39.
# sig baslik + bolmesiz toplam + clip 200 + cepte_prot + zor K=60 + temas maskesi + mol marji + poz marji.
MODELS.append(("BRL4X3_110k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BRL4X3/PLANET.iter-110000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("BRL4X3_35k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BRL4X3/PLANET.iter-35000"), "chemutils_esm2_vector", "v2", True))
# 2 Eki: BINDER POZ ARTIRMA kollari (gercek etiket, byer LMDB, Anzu GPU4/5, 140k). Indeks 40-41.
# Egitim modulu PLANET_model_esm2_vg_orj_gercek = ceptemask'in kopyasi, ileri gecis BIREBIR (fark yalniz
# compute_loss), layers ayni (md5 687ed1c6) -> ceptemask modulu ile yuklenir. ckpt md5 b71cc94b / afe885c6.
MODELS.append(("BYER0_140k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER0_140k/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("BYER05_140k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER05_140k/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
# 3 Eki: KONT (BYER + kontrastif InfoNCE) 140k. ckpt'ten yalniz prolig.kont_proj.* (kontrastif z, tahmine girmez)
# ayiklandi; tam KONT modeliyle ileri gecis BIREBIR (fark 0.00). ckpt md5 24a1a765 / 15e4a9ce (kaynak 57586c95 / ebe288fd).
MODELS.append(("KONT0_140k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/KONT0_140k/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("KONT05_140k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/KONT05_140k/PLANET.iter-140000"), "chemutils_esm2_vector", "v2", True))
# 5 Eki: BYER0/05 140k->500k devam (Anzu, LR/optimizer korunarak). md5 d0373f78 / 63b2c5bb.
MODELS.append(("BYER0_500k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER0_500k/PLANET.iter-500000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("BYER05_500k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER05_500k/PLANET.iter-500000"), "chemutils_esm2_vector", "v2", True))
# 5 Eki: 300k ara nokta (140k/300k/500k egrisi; 700k karari icin). md5 13fff405 / 040c4717.
MODELS.append(("BYER0_300k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER0_300k/PLANET.iter-300000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("BYER05_300k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER05_300k/PLANET.iter-300000"), "chemutils_esm2_vector", "v2", True))
# 7 Eki: KONT0/05 500k (Anzu, kont_proj ayiklandi, ileri gecis birebir). md5 4c5177c4 / 875c5465.
MODELS.append(("KONT0_500k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/KONT0_500k/PLANET.iter-500000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("KONT05_500k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/KONT05_500k/PLANET.iter-500000"), "chemutils_esm2_vector", "v2", True))
# 8 Eki: BYER0/05 700k (Anzu ham ckpt). md5 3cb31a89 / dd0141f7.
MODELS.append(("BYER0_700k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER0_700k/PLANET.iter-700000"), "chemutils_esm2_vector", "v2", True))
MODELS.append(("BYER05_700k", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/BYER05_700k/PLANET.iter-700000"), "chemutils_esm2_vector", "v2", True))

# ───────────────────────── mol2 / rmsd okuma (eski scriptle ayni) ─────────────────────────
def read_multi_mol2(path):
    from rdkit import Chem
    out, blocks, cur = [], [], None
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                if cur is not None:
                    blocks.append(cur)
                cur = [line]
            elif cur is not None:
                cur.append(line)
    if cur is not None:
        blocks.append(cur)
    for b in blocks:
        name = b[1].strip() if len(b) > 1 else ""
        txt = "".join(b)
        m = Chem.MolFromMol2Block(txt, removeHs=False, sanitize=True)
        if m is None:
            m = Chem.MolFromMol2Block(txt, removeHs=False, sanitize=False)
            if m is not None:
                try:
                    Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    m = None
        if m is not None and m.GetNumConformers() > 0:
            out.append((name, m))
    return out


def read_poses_template(path):
    """25 Eyl duzeltmesi: poza kor model ayni molekulun pozlarina farkli skor veriyordu ->
    pozlar mol2'den farkli KIMYAYLA okunuyordu. Burada grafik (atom/bag/yuk/aromatiklik)
    TEK SABLONDAN alinir (sanitize ile okunan ilk poz), her poza yalniz koordinatlari
    kopyalanir. Koruma: element dizisi sablonla ayni + sablon baglarinin uzunlugu 0,8-2,3 A.
    Doner: [(isim, mol)], atilan_sayisi"""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    raw = []
    blocks, cur = [], None
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                if cur is not None:
                    blocks.append(cur)
                cur = [line]
            elif cur is not None:
                cur.append(line)
    if cur is not None:
        blocks.append(cur)
    tmpl = None
    for b in blocks:
        name = b[1].strip() if len(b) > 1 else ""
        txt = "".join(b)
        m0 = Chem.MolFromMol2Block(txt, removeHs=False, sanitize=False)
        if m0 is None or m0.GetNumConformers() == 0:
            raw.append((name, None)); continue
        raw.append((name, m0))
        if tmpl is None:
            ms = Chem.MolFromMol2Block(txt, removeHs=False, sanitize=True)
            if ms is not None and ms.GetNumConformers() > 0:
                tmpl = ms
    if tmpl is None:
        return [], len(blocks)
    els = [a.GetSymbol() for a in tmpl.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in tmpl.GetBonds()]
    out, dropped = [], 0
    for name, m0 in raw:
        if m0 is None or m0.GetNumAtoms() != len(els) or \
           [a.GetSymbol() for a in m0.GetAtoms()] != els:
            dropped += 1; continue
        xyz = m0.GetConformer().GetPositions()
        bl = np.linalg.norm(xyz[[i for i, _ in bonds]] - xyz[[j for _, j in bonds]], axis=1) if bonds else np.array([1.5])
        if bl.min() < 0.8 or bl.max() > 2.3:
            dropped += 1; continue
        m = Chem.Mol(tmpl)
        conf = m.GetConformer()
        for k, (x, y, z) in enumerate(xyz):
            conf.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
        out.append((name, m))
    return out, dropped


def read_rmsd(path):
    d = {}
    with open(path, errors="ignore") as f:
        for i, line in enumerate(f):
            p = line.split()
            if i == 0 or len(p) < 2:
                continue
            try:
                d[p[0].strip()] = float(p[1])
            except ValueError:
                continue
    return d


# ───────────────────────── skorlama ─────────────────────────
def move_to(d, device):
    import torch
    if isinstance(d, torch.Tensor):
        return d.to(device)
    if isinstance(d, (list, tuple)):
        m = [move_to(x, device) for x in d]
        return tuple(m) if isinstance(d, tuple) else m
    return d


def score_full(model, pocket, cu, ligs, device, chunk=16):
    """planet_score.score_ligands ile AYNI birlestirme (cep gercekten cogaltilir, ligandlar
    tek cagrida); fark: her ligandin TAM temas vektorunu [atom*rezidu] dondurur."""
    import torch
    n_res = len(pocket.pocket_residues)
    affs, cmaps = [], []
    for i in range(0, len(ligs), chunk):
        part = ligs[i:i + chunk]
        res_f = cu.tensorize_protein_pocket([pocket] * len(part))
        # auto_detect=False: kiralite 3B'den YENIDEN hesaplanmasin (main'de sabitlendi)
        mol_f = cu.mol_batch_to_graph([l.mol for l in part], auto_detect=False)
        with torch.no_grad():
            preds = model(move_to(res_f, device), move_to(mol_f, device))
        affs.append(preds[2].detach().cpu().numpy().ravel())
        con = preds[1].detach().cpu().numpy().ravel()
        off = 0
        for l in part:
            na = l.atom_count
            cmaps.append(con[off:off + na * n_res].reshape(na, n_res))
            off += na * n_res
        if off != con.size:
            raise RuntimeError("temas vektoru boyu tutmuyor: %d != %d" % (off, con.size))
    return np.concatenate(affs), cmaps


def residue_atoms(pocket):
    xyz, idx = [], []
    for j, r in enumerate(pocket.pocket_residues):
        c = np.asarray(r.coordinates, dtype=float).reshape(-1, 3)
        xyz.append(c)
        idx.append(np.full(len(c), j))
    return np.concatenate(xyz), np.concatenate(idx), len(pocket.pocket_residues)


def geom_contact(mol, res_xyz, res_idx, n_res):
    """egitim etiketiyle ayni kural: atom, rezidunun HERHANGI bir atomuna <=4.0 A ise 1."""
    lig = mol.GetConformer().GetPositions()
    d = np.sqrt(((lig[:, None, :] - res_xyz[None, :, :]) ** 2).sum(-1))
    c = np.zeros((lig.shape[0], n_res), dtype=np.float32)
    close = d <= CONTACT_CUT
    for j in range(n_res):
        c[:, j] = close[:, res_idx == j].any(axis=1)
    return c


def agreement(p, c):
    """HAM ortalama log-olabilirlik. 25 Eyl duman testinde YANLI cikti: temas az olan pozlari
    oduller (poza kor modelde bile kotu pozlar daha yuksek). Seffaflik icin raporlanir, birincil degil."""
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return float(np.mean(c * np.log(p) + (1 - c) * np.log(1 - p)))


def dice(p, c):
    """BIRINCIL (duman testinden sonra secildi, sonucta acikca yazilacak): yumusak Dice
    2*sum(p*c) / (sum(p)+sum(c)) — tahmin edilen ve pozun gercek temaslarinin ortusmesi;
    fazla temas da eksik temas da cezalanir, sinif dengesizligine duyarsiz."""
    p = p.astype(float)
    den = p.sum() + c.sum()
    return float(2.0 * (p * c).sum() / den) if den > 0 else float("nan")


def balanced_ll(p, c):
    """IKINCIL: temas olan ve olmayan hucrelerin log-olabilirlik ortalamalarinin ortalamasi."""
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    pos, neg = c > 0.5, c <= 0.5
    if not pos.any() or not neg.any():
        return float("nan")
    return float(0.5 * (np.log(p[pos]).mean() + np.log(1 - p[neg]).mean()))


# ───────────────────────── ozet (beraberlige dayanikli) ─────────────────────────
def summarize(per, channel, seed=0):
    from scipy.stats import spearmanr
    from math import comb
    rng = np.random.default_rng(seed)
    tops = np.zeros(3); chance = np.zeros(3); n = 0; ties = 0
    rhos, nat_rank = [], []
    for pid, v in per.items():
        if "error" in v:
            continue
        s = np.array(v[channel], dtype=float); r = np.array(v["rmsd"], dtype=float)
        ok = np.isfinite(s) & np.isfinite(r); s, r = s[ok], r[ok]
        s = np.round(s, 5)   # parti bilesiminden gelen ~1e-7 sayisal gurultu beraberligi bozmasin
        if s.size < 5:
            continue
        n += 1
        good = r <= RMSD_CUT
        if np.ptp(s) == 0:
            ties += 1
        acc = np.zeros(3)
        for _ in range(N_TIE):
            order = np.lexsort((rng.random(s.size), -s))   # once skor, beraberlikte rastgele
            for k in range(3):
                acc[k] += good[order[:k + 1]].any()
        tops += acc / N_TIE
        g, N = int(good.sum()), s.size
        for k in range(3):
            chance[k] += 1 - comb(N - g, k + 1) / comb(N, k + 1)
        if s.std() > 0 and r.std() > 0:
            rhos.append(spearmanr(s, r).correlation)
        nat = int(np.argmin(r))
        nat_rank.append(int((s > s[nat]).sum()) + 1 + 0.5 * int((s == s[nat]).sum() - 1))
    f = lambda x: round(100.0 * x / n, 2) if n else None
    return {"n_targets": n, "top1": f(tops[0]), "top2": f(tops[1]), "top3": f(tops[2]),
            "sans_top1": f(chance[0]), "sans_top2": f(chance[1]), "sans_top3": f(chance[2]),
            "tum_berabere_orani": round(ties / n, 3) if n else None,
            "rho_vs_rmsd": round(float(np.mean(rhos)), 3) if rhos else None,
            "native_mean_rank": round(float(np.mean(nat_rank)), 1) if nat_rank else None}


def load_model(mod, moddir, ckpt, device):
    import torch
    if moddir and moddir not in sys.path:
        sys.path.insert(0, moddir)
    M = importlib.import_module(mod)
    print("  model modulu:", M.__file__, flush=True)
    m = M.PLANET(512, 8, 128, 128, 3, 10, 1, device).to(device)
    ck = torch.load(ckpt, map_location=device)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    miss, unexp = m.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit("agirlik uyusmazligi: eksik=%s beklenmeyen=%s" % (list(miss)[:5], list(unexp)[:5]))
    m.eval()
    return m


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
            cep_blosum(obj)
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


# 14 Eki: kontrastif 700k docking power (HIBIT tablosu). AYIKLANMIS ckpt SART (ham 700k prolig.kont_proj.* tasiyor).
MODELS.append(("KONT0_700k_ayik", "PLANET_model_esm2_vg_orj_ceptemask", os.path.join(ROOT, "birlesik"),
               os.path.join(SUMAFF, "ckpt/KONT0_700k_ayik/PLANET.iter-700000"), "chemutils_esm2_vector", "v2", True))

# 14 Eki: poza KOR kol docking power (Tablo 1 all-tied hucresi icin dogrudan kanit).
MODELS.append(("ABL2_P0_700k", "PLANET_model_esm2_uw_orj_gercek", "/arf/scratch/hnaydin/PLANET_ORIGINAL/anzu_kod_ikinci/ablasyon",
               os.path.join(SUMAFF, "ckpt/ABL2_P0_700k/PLANET.iter-700000"), "chemutils_esm2_vector", "v2", True))

MODELS.append(("ABL1_P0_700k", "PLANET_model_blosum_uw_orj_gercek",
               "/arf/scratch/hnaydin/PLANET_ORIGINAL/anzu_kod_ikinci/ablasyon",
               os.path.join(SUMAFF, "ckpt/ABL1_P0_700k/PLANET.iter-700000"),
               "chemutils_esm2_vector", "v2", False))   # 15 Eki: PLANET_BLOSUM=1 ile kosulur

if __name__ == "__main__":
    main()
