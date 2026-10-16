#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LITPCBA_dualchannel.py
======================
LIT-PCBA protokolu DEGISMEDI: 124 (target, template) kaydi, her kayitta o hedefin
TUM aktifleri ve TUM decoy'lari, AUC/EF kayit ICINDE hesaplanir, sonra ortalanir.
LigandDataset / collate_ligands / LitPcbaLmdbReader dogrudan mevcut scriptten
import edilir (kopyalanmaz) -> sapma riski yok.

Iki fark:
  1) --model_module ile HERHANGI bir model modulu kurulabilir (baseline+GN icin
     PLANET_model_gatenorm). Eski script --model_type noesm deyince eski
     PLANET_model'i kuruyordu; yanlis mimari + strict=False = sessiz bos model.
  2) Ayni ileri gecisten hem afinite (preds[2]) hem temas (preds[1]) skoru.
"""
import os, sys, math, argparse, importlib, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

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
    return h.LigandDataset, h.collate_ligands, h.LitPcbaLmdbReader


def calc_ef(active_scores, decoy_scores):
    """DUDE/LITPCBA scriptlerindeki calculate_ef ile birebir ayni formul."""
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


def predict_dual(model, loader, p_fres, p_alpha, device, center_ligands):
    aff_out, con_top10, con_mean = [], [], []
    n_res = p_fres.shape[0]
    pocket_center = p_alpha.mean(dim=0)
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
                preds = model((p_fres.repeat(bs, 1), None, res_scope, p_alpha.repeat(bs, 1)),
                              (fatoms, fbonds, agraph, bgraph, lig_scope, lig_coords))
                contacts = preds[1].detach().float().cpu().numpy().ravel()
                aff_out.append(preds[2].detach().float().cpu().numpy().ravel())
                off = 0
                for (_, n_atoms) in lig_scope:
                    n_pts = int(n_atoms) * int(n_res)
                    seg = contacts[off:off + n_pts]; off += n_pts
                    if seg.size == 0:
                        con_top10.append(np.nan); con_mean.append(np.nan)
                    else:
                        k = max(1, min(10, seg.size))
                        con_top10.append(float(np.sort(seg)[-k:].mean()))
                        con_mean.append(float(seg.mean()))
            except Exception as ex:
                print("    inference hatasi: %s" % ex, flush=True)
    if not aff_out:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(aff_out),
            np.asarray(con_top10, dtype=float),
            np.asarray(con_mean, dtype=float))


def metrics(act, dec):
    act = act[np.isfinite(act)]; dec = dec[np.isfinite(dec)]
    if len(act) == 0 or len(dec) == 0:
        return None
    y_true = np.concatenate([np.ones_like(act), np.zeros_like(dec)])
    y_score = np.concatenate([act, dec])
    if len(np.unique(y_score)) < 2:
        return None
    pr_, rc_, _ = precision_recall_curve(y_true, y_score)
    return {"AUC": roc_auc_score(y_true, y_score), "PR_AUC": auc(rc_, pr_),
            "EF": calc_ef(act, dec)}


def load_model_strict(mod_name, ckpt_path, a, device):
    PLANET = importlib.import_module(mod_name).PLANET
    model = PLANET(a.feature_dims, a.nheads, a.key_dims, a.value_dims,
                   a.pro_update_inters, a.lig_update_iters,
                   a.pro_lig_update_iters, device).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("!! state_dict uyusmazligi -- eksik:%d beklenmeyen:%d" % (len(missing), len(unexpected)))
        for k in list(missing)[:10]:    print("   eksik:", k)
        for k in list(unexpected)[:10]: print("   beklenmeyen:", k)
        raise SystemExit("Agirliklar tam yuklenemedi, is durduruldu.")
    print("Checkpoint yuklendi (tam eslesme): %s  total_step=%s"
          % (os.path.basename(ckpt_path), ck.get("total_step", "?") if isinstance(ck, dict) else "?"),
          flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_module", required=True)
    ap.add_argument("--lmdb_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--limit_targets", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--code_paths", required=True)
    ap.add_argument("--helper_module", required=True)
    ap.add_argument("--model_type", default="noesm")
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
    LigandDataset, collate_ligands, LitPcbaLmdbReader = _bind(a.code_paths, a.helper_module)
    device = torch.device(a.device)
    print("Cihaz: %s | model: %s | yardimci: %s" % (device, a.model_module, a.helper_module), flush=True)
    model = load_model_strict(a.model_module, a.model_path, a, device)

    reader = LitPcbaLmdbReader(a.lmdb_dir)
    rows = []
    if os.path.exists(a.out_csv):
        try:
            rows = pd.read_csv(a.out_csv).to_dict("records")
            print("Mevcut CSV: %d kayit, devam ediliyor." % len(rows), flush=True)
        except Exception:
            rows = []
    completed = {r["Run_ID"] for r in rows}

    t0 = time.time(); done = 0
    for tidx, rec in enumerate(reader):
        if a.nshards > 1 and (tidx % a.nshards) != a.shard:
            continue
        rid = rec.get("run_id") or rec.get("key")
        if rid in completed:
            continue
        if a.limit_targets and done >= a.limit_targets:
            break
        pocket = reader.load(rec.get("key") or rid)
        if pocket is None:
            print("  LMDB'de yok: %s" % rid, flush=True); continue
        p_fres = torch.from_numpy(pocket.res_features).float().to(device)
        p_alpha = torch.from_numpy(pocket.alpha_coordinates).float().to(device)

        act_ds = LigandDataset(rec["actives_file"])
        dec_ds = LigandDataset(rec["decoys_file"])
        if len(act_ds) == 0 or len(dec_ds) == 0:
            print("  Ligand yok: %s" % rid, flush=True); continue
        al = DataLoader(act_ds, batch_size=a.batch_size, collate_fn=collate_ligands, num_workers=0)
        dl = DataLoader(dec_ds, batch_size=a.batch_size, collate_fn=collate_ligands, num_workers=0)

        a_aff, a_c10, a_cme = predict_dual(model, al, p_fres, p_alpha, device, a.center_ligands)
        d_aff, d_c10, d_cme = predict_dual(model, dl, p_fres, p_alpha, device, a.center_ligands)
        if len(a_aff) == 0 or len(d_aff) == 0:
            print("  Skor yok: %s" % rid, flush=True); continue

        m_aff = metrics(a_aff, d_aff); m_c10 = metrics(a_c10, d_c10); m_cme = metrics(a_cme, d_cme)
        if m_aff is None:
            print("  Metrik yok: %s" % rid, flush=True); continue

        row = {"Run_ID": rid, "Target": rec.get("target_name", ""),
               "Template": rec.get("template_name", ""),
               "n_actives": int(len(a_aff)), "n_decoys": int(len(d_aff)),
               "AUC_afinite": m_aff["AUC"], "EF1_afinite": m_aff["EF"]["EF_1.0%"],
               "AUC_temas_top10": (m_c10["AUC"] if m_c10 else np.nan),
               "EF1_temas_top10": (m_c10["EF"]["EF_1.0%"] if m_c10 else np.nan),
               "AUC_temas_ort": (m_cme["AUC"] if m_cme else np.nan),
               "EF1_temas_ort": (m_cme["EF"]["EF_1.0%"] if m_cme else np.nan),
               # [13 Eki 2026 YAMA] calc_ef zaten 0.5/1/2/5% hesapliyordu, yalniz 1% yaziliyordu.
               "EF05_afinite": m_aff["EF"]["EF_0.5%"], "EF2_afinite": m_aff["EF"]["EF_2.0%"],
               "EF5_afinite": m_aff["EF"]["EF_5.0%"]}
        # [13 Eki 2026 YAMA] molekul bazinda skorlari kaydet -> sonradan HERHANGI bir metrik
        # (genis EF, boyut duzeltmesi, zenginlestirme egrisi) yeniden kosmadan hesaplanabilsin.
        try:
            import os as _os, re as _re
            _d = _os.path.dirname(a.out_csv)
            _tag = _re.sub(r"_shard\d+\.csv$", "", _os.path.basename(a.out_csv))
            _gid = _re.sub(r"[^A-Za-z0-9_.-]", "_", str(rid))
            np.savez_compressed(_os.path.join(_d, "skor_%s_%s.npz" % (_tag, _gid)),
                                a_aff=a_aff, d_aff=d_aff, a_c10=a_c10, d_c10=d_c10,
                                a_cme=a_cme, d_cme=d_cme)
        except Exception as _e:
            print("  npz yazilamadi (%s): %s" % (rid, _e), flush=True)
        rows.append(row); done += 1
        print("  %-22s | AFN AUC:%.4f EF1:%5.2f | TMS AUC:%.4f | act:%d dec:%d | %.1f dk"
              % (rid, row["AUC_afinite"], row["EF1_afinite"], row["AUC_temas_top10"],
                 row["n_actives"], row["n_decoys"], (time.time() - t0) / 60.0), flush=True)
        pd.DataFrame(rows).to_csv(a.out_csv, index=False)

    if rows:
        df = pd.DataFrame(rows)
        print("\n===== OZET [%s] : %d kayit =====" % (a.run_name, len(df)))
        print("  Afinite kanali : Mean AUC=%.4f  Mean EF1%%=%.2f" % (df.AUC_afinite.mean(), df.EF1_afinite.mean()))
        print("  Temas (top10)  : Mean AUC=%.4f  Mean EF1%%=%.2f" % (df.AUC_temas_top10.mean(), df.EF1_temas_top10.mean()))
        print("  CSV: %s" % a.out_csv)


if __name__ == "__main__":
    main()
