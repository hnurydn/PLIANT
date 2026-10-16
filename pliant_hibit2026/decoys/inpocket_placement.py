# -*- coding: utf-8 -*-
"""Decoy yerleştirme v3 (19 Eyl 2026): çakışma ATOM YARIÇAPLARINA göre, temas ödülü kısılmış.
v2'de tek 3,0 Å eşiği kullanılıyordu; çakışma cezası binder seviyesine indi ama Vina %77 itici buldu.
v3: her atom çifti için izinli mesafe = F * (r_i + r_j) (van der Waals yarıçapları).
E = Σ_pairs max(0, F*(ri+rj) - d)  -  ALFA * temas  +  LAMBDA * max(0, |merkez-ref| - RMAX)
"""
import numpy as np

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75,
       "BR": 1.85, "I": 1.98, "H": 1.20, "SE": 1.90, "ZN": 1.39, "FE": 1.40, "MG": 1.73,
       "CA": 1.95, "MN": 1.40, "CU": 1.40, "NA": 2.27, "K": 2.75}
VARSAYILAN = 1.70
LAMBDA = 50.0


def vdw_radius(elementler):
    return np.array([VDW.get(str(e).upper(), VARSAYILAN) for e in elementler], dtype=float)


def rastgele_donusler(k, rng):
    q = rng.normal(size=(k, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    if k: R[0] = np.eye(3)
    return R


def kucuk_donus(rng, sigma_derece):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
    th = np.deg2rad(rng.normal(0, sigma_derece))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def olc(x_agir, r_lig, cep, r_cep, F, temas_ust=1.45):
    """çakışma cezası (yarıçap tabanlı) ve temas sayısı."""
    d = np.sqrt(((x_agir[:, None, :] - cep[None, :, :]) ** 2).sum(-1))
    izin = F * (r_lig[:, None] + r_cep[None, :])
    cak = float(np.clip(izin - d, 0, None).sum())
    oran = d / (r_lig[:, None] + r_cep[None, :])
    tem = int(((oran >= F) & (oran <= temas_ust)).any(1).sum())   # temas kabuğunda atom sayısı
    return cak, tem


def place_in_pocket(x0, agir, r_lig_agir, cep, r_cep, ref, rng, F=0.85, ALFA=0.05, RMAX=3.0,
                 n_aday=200, n_baslangic=3, n_adim=60, jit=1.5):
    merkezli = x0 - x0.mean(0)
    R = rastgele_donusler(n_aday, rng)
    T = rng.uniform(-jit, jit, size=(n_aday, 3)); T[0] = 0.0
    hepsi = np.einsum("nij,aj->nai", R, merkezli) + (ref + T)[:, None, :]
    # ucuz ön seçim: yalnız min-mesafe tabanlı kaba ceza
    A = hepsi[:, agir, :]
    dmin = np.sqrt(((A[:, :, None, :] - cep[None, None, :, :]) ** 2).sum(-1)).min(2)
    kaba = np.clip(3.2 - dmin, 0, None).sum(1)
    sira = np.argsort(kaba)[:n_baslangic]

    def E(x):
        cak, tem = olc(x[agir], r_lig_agir, cep, r_cep, F)
        sap = float(np.linalg.norm(x.mean(0) - ref))
        return cak - ALFA * tem + LAMBDA * max(0.0, sap - RMAX), cak, tem, sap

    en_iyi = None
    for j in sira:
        Rj, Tj = R[j].copy(), T[j].copy()
        x = merkezli @ Rj.T + ref + Tj
        e, c, t, s = E(x)
        for it in range(n_adim):
            olcek = 1.0 - it / n_adim
            Rn = kucuk_donus(rng, 25.0 * olcek + 2.0) @ Rj
            Tn = Tj + rng.normal(0, 0.6 * olcek + 0.05, size=3)
            xn = merkezli @ Rn.T + ref + Tn
            en, cn, tn, sn = E(xn)
            if en < e:
                Rj, Tj, x, e, c, t, s = Rn, Tn, xn, en, cn, tn, sn
        if en_iyi is None or e < en_iyi[1]:
            en_iyi = (x.astype(np.float32), e, c, t, s)
    return en_iyi
