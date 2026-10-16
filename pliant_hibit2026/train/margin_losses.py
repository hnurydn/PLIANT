#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POZ MARJ KAYBI — afinite kanalina goreli (hinge) kisit (25 Eyl 2026)
=======================================================================
NEDEN
  Mevcut afinite kaybi `pK_flags` ile maskeli: decoy'lar afinite terimini HIC gormuyor,
  dolayisiyla kanalin pozlari siralamak icin bir sebebi yok. Mesafe taramasi (is 6409058)
  bunun sonucunu olctu: ortalama-toplamali kollarda ligand cepten cikarildikca tahmin
  edilen pK YUKSELIYOR (ESM_VG FIX 5,91 -> 8,80; v4P 6,33 -> 9,41). VGTOP (toplam) kaba
  fizigi duzeltiyor (5,43 -> 0,00) ama cep ICINDEKI ince poz ayrimini cozmuyor
  (afinite docking top1 18,7 < sans 24,4).

NE YAPAR
  Ayni molekul + ayni cep icin "kristal poz" ile "bozulmus poz" arasinda marj sartI:

      L_marj = mean_i  relu( m - ( y(kristal_i) - y(bozuk_i) ) )

  Yani mutlak bir etiket dayatmaz (decoy'un gercek afinitesini bilmiyoruz zaten);
  yalnizca "dogru poz, bozuk pozdan en az m kadar iyi olmali" der.
  Literatur karsiligi: PIGNet'in hinge'i (decoy >= 1 kcal/mol kotu) ve Interformer'in
  kontrastif kaybi (rho(afinite,RMSD) -0,174 -> -0,562).

⭐ TASARIM KARARI — negatif poz CEP ICINDE uretilir
  Bozulma = agirlik merkezi etrafinda RASTGELE DONME + KUCUK oteleme (varsayilan 1-3 A).
  Uzaga otelemek (8+ A) BILEREK yapilmaz: mesafe taramasi, modelin "uzaktaysa temas yok"
  kisayolunu zaten ogrendigini gosterdi; uzak negatif bu kisayoli pekistirir, yeni bir sey
  ogretmez. Cep icinde donmus poz ise mesafe dagilimini buyuk olcude korur, dolayisiyla
  ancak YON + SEKIL TAMAMLAYICILIGI ile ayirt edilebilir.

  Donme HAAR-duzgun ve det(R) = +1 (oz donme) -> kiralite korunur. Atom ozellikleri
  (kiralite etiketi dahil) veri hazirliginda hesaplandigi icin degismez; degisen tek sey
  `lig_coordinates`, yani modelin geo_bias kanalina giren ligand-rezidu geometrisi.

MODEL SINIFI DEGISMEZ
  Yeni parametre (ornegin 4. bir log_sigma) EKLENMEZ -> checkpoint anahtarlari ayni kalir,
  mevcut degerlendirme scriptleri (eval_docking_power / eval_contact_auroc / casf_afinite_kristal /
  eval_mesafe_taramasi) hicbir degisiklik olmadan bu kolu okur. Marj terimi sabit agirlikla
  (lambda) toplam kayba eklenir.
"""
import torch


def _haar_rotasyon(gen, device, dtype):
    """det=+1 rastgele donme matrisi (QR + isaret duzeltmesi). Yansima YOK -> kiralite korunur."""
    A = torch.randn(3, 3, generator=gen, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)      # QR isaret belirsizligini kaldir
    if torch.det(Q) < 0:                                     # yansimayi oz donmeye cevir
        Q[:, 0] = -Q[:, 0]
    return Q


def bozuk_lig_koordinatlari(lig_coords, lig_scope, gen,
                            oteleme_min=1.0, oteleme_max=3.0, sec_maske=None):
    """Her molekul icin BAGIMSIZ kati-cisim bozulmasi (merkez etrafinda donme + oteleme).

    lig_coords : [toplam_atom, 3]
    lig_scope  : [(baslangic_atom, atom_sayisi), ...]  (molekul basina)
    sec_maske  : None ya da molekul basina bool listesi; False olanlar DOKUNULMADAN birakilir
    Donus      : ayni sekilde yeni tensor (girdi degistirilmez)
    """
    yeni = lig_coords.clone()
    for j, (bas, n) in enumerate(lig_scope):
        if sec_maske is not None and not bool(sec_maske[j]):
            continue
        if n < 2:
            continue
        P = yeni[bas:bas + n]
        merkez = P.mean(dim=0, keepdim=True)
        R = _haar_rotasyon(gen, P.device, P.dtype)
        # oteleme: rastgele yon, [oteleme_min, oteleme_max] araliginda buyukluk
        v = torch.randn(3, generator=gen, device=P.device, dtype=P.dtype)
        v = v / (v.norm() + 1e-8)
        u = torch.rand(1, generator=gen, device=P.device, dtype=P.dtype)
        d = oteleme_min + (oteleme_max - oteleme_min) * u
        yeni[bas:bas + n] = (P - merkez) @ R.transpose(0, 1) + merkez + (v * d).unsqueeze(0)
    return yeni


def pose_margin_loss(model, res_batch, mol_batch, targets, gen, tahmin_poz,
               marj=1.0, oteleme_min=1.0, oteleme_max=3.0, yalniz_binder=True):
    """Kristal poz ile bozulmus poz arasinda hinge. (kayip, ortalama_fark, ihlal_orani) doner.

    tahmin_poz: ANA ileri gecisin afinite ciktisi (predictions[2]). Disaridan alinir ki
    ayni hesap iki kez yapilmasin -> adim maliyeti +1 ileri gecis (2 degil).

    Secim: yalniz_binder=True ise yalnizca complex_label==1 VE pK_flag==1 olan molekuller
    (yani gercek kristal pozu olan binder'lar) kullanilir. Decoy'lar disarida birakilir —
    onlarin "dogru poz"u yok, dolayisiyla marj tanimsiz olurdu.
    """
    if len(targets) == 6:
        _, _, _, pK_flags, complex_labels, _ = targets
    else:
        _, _, _, pK_flags, complex_labels = targets

    lig_scope = mol_batch[4]
    n_mol = len(lig_scope)
    bayrak = pK_flags.detach().reshape(-1).tolist()
    etiket = [int(x.item()) if torch.is_tensor(x) else int(x) for x in complex_labels]
    if yalniz_binder:
        sec = [(etiket[j] == 1 and bayrak[j] >= 0.5) for j in range(n_mol)]
    else:
        sec = [True] * n_mol
    if not any(sec):
        sifir = torch.zeros((), device=pK_flags.device)
        return sifir, sifir, sifir

    bozuk_coords = bozuk_lig_koordinatlari(mol_batch[5], lig_scope, gen,
                                           oteleme_min, oteleme_max, sec_maske=sec)
    bozuk_batch = (mol_batch[0], mol_batch[1], mol_batch[2], mol_batch[3], lig_scope, bozuk_coords)

    y_poz = tahmin_poz.reshape(-1)                      # ana gecisten geldi, yeniden hesaplanmaz
    y_neg = model(res_batch, bozuk_batch)[2].reshape(-1)

    idx = torch.tensor([j for j in range(n_mol) if sec[j]], device=y_poz.device, dtype=torch.long)
    fark = y_poz[idx] - y_neg[idx]
    kayip = torch.clamp(marj - fark, min=0.0).mean()
    return kayip, fark.mean().detach(), (fark < marj).float().mean().detach()


def molecule_margin_loss(tahmin_afinite, targets, marj=1.0):
    """MOLEKUL marji: ayni cep grubunda  y(binder) >= y(decoy) + marj.

    Poz marjindan FARKI: orada ayni molekulun dogru/bozuk POZU kiyaslaniyordu; burada ayni
    cebin BINDER'i ile DECOY MOLEKULLERI kiyaslaniyor. Ek ileri gecis GEREKMEZ — ana gecisin
    afinite ciktisi yeter (decoy'lar zaten partide).

    Literatur karsiligi: PIGNet'in esitsizlik kisiti (decoy, gercek yapidan en az bir marj
    kadar kotu olmali). Mutlak etiket YOK: decoy'un gercek afinitesini bilmiyoruz.
    """
    if len(targets) == 6:
        _, _, _, _, complex_labels, group_ids = targets
    else:
        _, _, _, _, complex_labels = targets
        group_ids = None
    y = tahmin_afinite.reshape(-1)
    n = y.shape[0]

    def _int(x):
        return int(x.item()) if torch.is_tensor(x) else int(x)

    etiket = [_int(x) for x in complex_labels][:n]
    if group_ids is not None:
        grup = [_int(x) for x in group_ids][:n]
    else:
        grup = [0] * n

    ciftler = []
    for g in set(grup):
        b = [j for j in range(n) if grup[j] == g and etiket[j] == 1]
        d = [j for j in range(n) if grup[j] == g and etiket[j] == 0]
        for bi in b:
            for di in d:
                ciftler.append((bi, di))
    if not ciftler:
        sifir = torch.zeros((), device=y.device)
        return sifir, sifir, sifir
    bi = torch.tensor([c[0] for c in ciftler], device=y.device, dtype=torch.long)
    di = torch.tensor([c[1] for c in ciftler], device=y.device, dtype=torch.long)
    fark = y[bi] - y[di]
    kayip = torch.clamp(marj - fark, min=0.0).mean()
    return kayip, fark.mean().detach(), (fark < marj).float().mean().detach()
