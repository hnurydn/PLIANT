#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[SKOR KOPYASI 18 Eyl 2026] bench19_code/DUDE_dualchannel.py ile ayni; TEK FARK: her hedef icin
ligand basi afinite/temas skorlari ve atom sayilari skor_<run>_<hedef>.npz olarak kaydedilir.

DUDE_dualchannel.py
===================
DUD-E protokolu DEGISMEDI: 101 hedef, her hedefte o hedefin TUM aktifleri ve
TUM decoy'lari, AUC/EF hedef ICINDE hesaplanir, sonra hedefler uzerinden ortalanir.
DUDE_benchmark_centered.py ile birebir ayni dongu, ayni calculate_ef, ayni
center_ligands mantigi.

TEK FARK: model 3 cikti donduruyor, eski kod ikisini atip sadece afiniteyi
tutuyordu. Bu surum AYNI ileri gecisten HEM afinite (preds[2]) HEM temas
(preds[1]) skorunu cikarir -> her hedef icin iki AUC. Ek maliyet YOK.

  afinite skoru = preds[2]                      (bugune kadar raporlanan)
  temas skoru   = preds[1] icindeki ligand-basi top-10 ortalamasi
                  (CASF cep-ici probe'uyla ayni tanim -> karsilastirilabilir)
"""
import os, sys, json, gzip, pickle, math, argparse, importlib, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# ---- yardimci moduller calisma aninda baglanir (TRUBA ve Anzu farkli isimler) ----
def _bind(code_paths, helper_module):
    for _p in reversed([x.strip() for x in code_paths.split(",") if x.strip()]):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    h = importlib.import_module(helper_module)
    # LMDB pickle'lari orijinal script __main__ iken yazilmis -> kayit siniflari
    # __main__'de bulunmali, yoksa unpickle "Can't get attribute" verir.
    import __main__, inspect
    for _n, _o in vars(h).items():
        if inspect.isclass(_o) and not hasattr(__main__, _n):
            setattr(__main__, _n, _o)
    return h.LigandDataset, h.collate_ligands, h.DudeLmdbReader


def calculate_ef(active_scores, decoy_scores):
    """DUDE scriptindeki calculate_ef ile birebir ayni formul."""
    y_true = np.concatenate([np.ones_like(active_scores), np.zeros_like(decoy_scores)])
    y_score = np.concatenate([active_scores, decoy_scores])
    sorted_labels = y_true[np.argsort(-y_score)]
    n_total, n_actives = len(y_score), len(active_scores)
    out = {}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        n = math.ceil(n_total * pct)
        out["EF_%.1f%%" % (pct * 100)] = ((sorted_labels[:n].sum() / n) / (n_actives / n_total)
                                          if n > 0 and n_actives > 0 else 0.0)
    return out


def predict_dual(model, loader, p_fresidues, p_alpha_coords, device, center_ligands):
    """Ayni ileri gecisten hem afinite hem temas skoru dondurur."""
    aff_out, con_top10, con_mean, natoms = [], [], [], []
    n_res = p_fresidues.shape[0]
    pocket_center = p_alpha_coords.mean(dim=0)
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            fatoms, fbonds, agraph, bgraph, lig_scope = batch[:5]
            lig_coords = batch[5] if len(batch) == 6 else torch.zeros(fatoms.shape[0], 3)
            fatoms = fatoms.to(device); fbonds = fbonds.to(device)
            agraph = agraph.to(device); bgraph = bgraph.to(device)
            lig_coords = lig_coords.to(device)
            if center_ligands:
                for start, length in lig_scope:
                    mc = lig_coords[start:start + length]
                    lig_coords[start:start + length] = mc - mc.mean(dim=0) + pocket_center
            bs = len(lig_scope)
            res_scope = [(i * n_res, n_res) for i in range(bs)]
            try:
                preds = model(
                    (p_fresidues.repeat(bs, 1), None, res_scope, p_alpha_coords.repeat(bs, 1)),
                    (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coords))
                contacts = preds[1].detach().float().cpu().numpy().ravel()
                aff      = preds[2].detach().float().cpu().numpy().ravel()
                aff_out.append(aff)
                natoms.extend(int(n) for (_, n) in lig_scope)
                off = 0
                for (_, n_atoms) in lig_scope:
                    n_pts = int(n_atoms) * int(n_res)
                    seg = contacts[off:off + n_pts]
                    off += n_pts
                    if seg.size == 0:
                        con_top10.append(np.nan); con_mean.append(np.nan)
                    else:
                        k = max(1, min(10, seg.size))
                        con_top10.append(float(np.sort(seg)[-k:].mean()))
                        con_mean.append(float(seg.mean()))
            except Exception as ex:
                print("    inference hatasi: %s" % ex, flush=True)
    if not aff_out:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(aff_out),
            np.asarray(con_top10, dtype=float),
            np.asarray(con_mean, dtype=float),
            np.asarray(natoms, dtype=int))


def metrics(act, dec):
    """Tek kanal icin hedef-ici AUC + EF (eski scriptteki hesabin aynisi)."""
    ok_a = np.isfinite(act); ok_d = np.isfinite(dec)
    act, dec = act[ok_a], dec[ok_d]
    if len(act) == 0 or len(dec) == 0:
        return None
    y_true  = np.concatenate([np.ones_like(act), np.zeros_like(dec)])
    y_score = np.concatenate([act, dec])
    if len(np.unique(y_score)) < 2:
        return None
    roc = roc_auc_score(y_true, y_score)
    pr_, rc_, _ = precision_recall_curve(y_true, y_score)
    return {"AUC": roc, "PR_AUC": auc(rc_, pr_), "EF": calculate_ef(act, dec)}


def load_model_strict(mod_name, ckpt_path, a, device):
    PLANET = importlib.import_module(mod_name).PLANET
    model = PLANET(a.feature_dims, a.nheads, a.key_dims, a.value_dims,
                   a.pro_update_inters, a.lig_update_iters,
                   a.pro_lig_update_iters, device).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False sessizce bos model birakabilir -> acikca dogrula
    if missing or unexpected:
        print("!! state_dict uyusmazligi -- eksik:%d beklenmeyen:%d" % (len(missing), len(unexpected)))
        for k in list(missing)[:10]:    print("   eksik:", k)
        for k in list(unexpected)[:10]: print("   beklenmeyen:", k)
        raise SystemExit("Agirliklar tam yuklenemedi, is durduruldu.")
    step = ck.get("total_step", "?") if isinstance(ck, dict) else "?"
    print("Checkpoint yuklendi (tam eslesme): %s  total_step=%s" % (os.path.basename(ckpt_path), step),
          flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_module", required=True,
                    help="ornek: PLANET_model_esm2_vector_gating_normalized_sumln_gn")
    ap.add_argument("--lmdb_dir",   required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv",    required=True)
    ap.add_argument("--run_name",   required=True)
    ap.add_argument("--limit_targets", type=int, default=0, help="0 = hepsi (duman testi icin 2-3 ver)")
    ap.add_argument("--shard",   type=int, default=0, help="bu gorevin parca indeksi (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=1, help="toplam parca sayisi")
    ap.add_argument("--code_paths", required=True, help="virgulle ayrilmis modul dizinleri")
    ap.add_argument("--helper_module", required=True,
                    help="TRUBA: DUDE_benchmark_centered | Anzu: DUDE_benchmark_unified")
    ap.add_argument("--model_type", default="noesm",
                    help="yalnizca yardimci modulun argv taramasini tatmin etmek icin")
    ap.add_argument("--center_ligands", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--feature_dims", type=int, default=512)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--key_dims", type=int, default=128)
    ap.add_argument("--value_dims", type=int, default=128)
    ap.add_argument("--pro_update_inters", type=int, default=3)
    ap.add_argument("--lig_update_iters", type=int, default=10)
    ap.add_argument("--pro_lig_update_iters", type=int, default=1)
    a = ap.parse_args()

    global LigandDataset, collate_ligands
    LigandDataset, collate_ligands, DudeLmdbReader = _bind(a.code_paths, a.helper_module)
    device = torch.device(a.device)
    print("Cihaz: %s | model: %s | yardimci: %s" % (device, a.model_module, a.helper_module), flush=True)
    model = load_model_strict(a.model_module, a.model_path, a, device)

    reader = DudeLmdbReader(a.lmdb_dir)
    rows, done = [], 0
    if os.path.exists(a.out_csv):
        try:
            rows = pd.read_csv(a.out_csv).to_dict("records")
            print("Mevcut CSV: %d hedef tamamlanmis, devam ediliyor." % len(rows), flush=True)
        except Exception:
            rows = []
    completed = {r["Target"] for r in rows}

    t0 = time.time()
    for tidx, rec in enumerate(reader):
        if a.nshards > 1 and (tidx % a.nshards) != a.shard:
            continue
        tname = rec["target_name"]
        if tname in completed:
            continue
        if a.limit_targets and done >= a.limit_targets:
            break
        pocket = reader.load(tname)
        if pocket is None:
            print("  LMDB'de yok: %s" % tname, flush=True); continue
        p_fres = torch.from_numpy(pocket.res_features).float().to(device)
        p_alpha = torch.from_numpy(pocket.alpha_coordinates).float().to(device)

        act_ds = LigandDataset(rec["actives_file"])
        dec_ds = LigandDataset(rec["decoys_file"])
        if len(act_ds) == 0 or len(dec_ds) == 0:
            print("  Ligand yok: %s" % tname, flush=True); continue

        al = DataLoader(act_ds, batch_size=a.batch_size, collate_fn=collate_ligands, num_workers=0)
        dl = DataLoader(dec_ds, batch_size=a.batch_size, collate_fn=collate_ligands, num_workers=0)

        a_aff, a_c10, a_cme, a_nat = predict_dual(model, al, p_fres, p_alpha, device, a.center_ligands)
        d_aff, d_c10, d_cme, d_nat = predict_dual(model, dl, p_fres, p_alpha, device, a.center_ligands)
        # [SKOR] 18 Eyl: boyut duzeltmeli analiz icin ligand basi skorlar + atom sayisi
        np.savez_compressed(os.path.join(os.path.dirname(a.out_csv), "skor_%s_%s.npz" % (a.run_name.split("_shard")[0], tname)),
                            a_aff=a_aff, d_aff=d_aff, a_c10=a_c10, d_c10=d_c10, a_nat=a_nat, d_nat=d_nat)
        if len(a_aff) == 0 or len(d_aff) == 0:
            print("  Skor yok: %s" % tname, flush=True); continue

        m_aff = metrics(a_aff, d_aff)
        m_c10 = metrics(a_c10, d_c10)
        m_cme = metrics(a_cme, d_cme)
        if m_aff is None:
            print("  Metrik hesaplanamadi: %s" % tname, flush=True); continue

        row = {"Target": tname, "n_actives": int(len(a_aff)), "n_decoys": int(len(d_aff)),
               "AUC_afinite": m_aff["AUC"], "EF1_afinite": m_aff["EF"]["EF_1.0%"],
               "AUC_temas_top10": (m_c10["AUC"] if m_c10 else np.nan),
               "EF1_temas_top10": (m_c10["EF"]["EF_1.0%"] if m_c10 else np.nan),
               "AUC_temas_ort": (m_cme["AUC"] if m_cme else np.nan),
               "EF1_temas_ort": (m_cme["EF"]["EF_1.0%"] if m_cme else np.nan)}
        rows.append(row); done += 1
        print("  %-18s | AFN AUC:%.4f EF1:%5.2f | TMS AUC:%.4f EF1:%5.2f | act:%d dec:%d | %.1f dk"
              % (tname, row["AUC_afinite"], row["EF1_afinite"],
                 row["AUC_temas_top10"], row["EF1_temas_top10"],
                 row["n_actives"], row["n_decoys"], (time.time() - t0) / 60.0), flush=True)
        pd.DataFrame(rows).to_csv(a.out_csv, index=False)

    if rows:
        df = pd.DataFrame(rows)
        print("\n===== OZET [%s] : %d hedef =====" % (a.run_name, len(df)))
        print("  Afinite kanali (bugune kadar raporlanan): Mean AUC=%.4f  Mean EF1%%=%.2f"
              % (df.AUC_afinite.mean(), df.EF1_afinite.mean()))
        print("  Temas kanali (top10)                    : Mean AUC=%.4f  Mean EF1%%=%.2f"
              % (df.AUC_temas_top10.mean(), df.EF1_temas_top10.mean()))
        print("  Temas kanali (ortalama)                 : Mean AUC=%.4f  Mean EF1%%=%.2f"
              % (df.AUC_temas_ort.mean(), df.EF1_temas_ort.mean()))
        print("  CSV: %s" % a.out_csv)


if __name__ == "__main__":
    main()
