#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MARGIN LOSSES - relative (hinge) constraints on the affinity channel (25 Sep 2026)
=====================================================================================
WHY
  The affinity loss is masked by `pK_flags`, so decoys never reach the affinity term and the
  channel has no reason to rank poses at all. A distance sweep measured the consequence: in the
  mean-pooled arms the predicted pK RISES as the ligand is pulled out of the pocket (ESM_VG
  5.91 -> 8.80; v4P 6.33 -> 9.41). The sum-pooled arm fixes the coarse physics (5.43 -> 0.00)
  but does not solve the fine pose discrimination INSIDE the pocket (affinity docking top1 18.7,
  below the 24.4 chance level).

WHAT IT DOES
  For the same molecule in the same pocket, it requires a margin between the crystal pose and a
  perturbed pose:

      L_margin = mean_i  relu( m - ( y(crystal_i) - y(perturbed_i) ) )

  No absolute label is imposed - we do not know a decoy's true affinity anyway. The constraint
  only says that the correct pose must score at least m better than the perturbed one.
  Related work: the hinge in PIGNet (a decoy must be at least 1 kcal/mol worse) and the
  contrastive loss in Interformer (rho(affinity, RMSD) -0.174 -> -0.562).

DESIGN DECISION - the negative pose is generated INSIDE the pocket
  The perturbation is a RANDOM ROTATION about the centre of mass plus a SMALL translation
  (1-3 A by default). Translating far away (8+ A) is deliberately avoided: the distance sweep
  showed the model has already learned the "far away means no contact" shortcut, and a distant
  negative only reinforces it instead of teaching anything new. A rotated pose inside the pocket
  largely preserves the distance distribution, so it can only be told apart through ORIENTATION
  and SHAPE COMPLEMENTARITY.

  The rotation is Haar-uniform with det(R) = +1 (a proper rotation), so chirality is preserved.
  Atom features are unchanged.
"""
import torch


def _haar_rotasyon(gen, device, dtype):
    """Random rotation matrix with det = +1 (QR plus a sign fix). No reflection, so chirality is preserved."""
    A = torch.randn(3, 3, generator=gen, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)      # remove the QR sign ambiguity
    if torch.det(Q) < 0:                                     # turn a reflection into a proper rotation
        Q[:, 0] = -Q[:, 0]
    return Q


def bozuk_lig_koordinatlari(lig_coords, lig_scope, gen,
                            oteleme_min=1.0, oteleme_max=3.0, sec_maske=None):
    """An INDEPENDENT rigid-body perturbation for each molecule (rotation about the centre plus a translation).

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
        # translation: random direction, magnitude in [oteleme_min, oteleme_max]
        v = torch.randn(3, generator=gen, device=P.device, dtype=P.dtype)
        v = v / (v.norm() + 1e-8)
        u = torch.rand(1, generator=gen, device=P.device, dtype=P.dtype)
        d = oteleme_min + (oteleme_max - oteleme_min) * u
        yeni[bas:bas + n] = (P - merkez) @ R.transpose(0, 1) + merkez + (v * d).unsqueeze(0)
    return yeni


def pose_margin_loss(model, res_batch, mol_batch, targets, gen, tahmin_poz,
               marj=1.0, oteleme_min=1.0, oteleme_max=3.0, yalniz_binder=True):
    """Hinge between the crystal pose and the perturbed pose.

    Returns (loss, mean_difference, violation_rate).

    tahmin_poz: the affinity output of the MAIN forward pass (predictions[2]). It is passed in
    so the same computation is not repeated, which keeps the step cost at one extra forward
    pass rather than two.

    Selection: with yalniz_binder=True only molecules with complex_label == 1 AND pK_flag == 1
    are used, i.e. binders that have a real crystal pose. Decoys are left out because they have
    no "correct pose", so the margin would be undefined for them.
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

    y_poz = tahmin_poz.reshape(-1)                      # comes from the main pass; not recomputed
    y_neg = model(res_batch, bozuk_batch)[2].reshape(-1)

    idx = torch.tensor([j for j in range(n_mol) if sec[j]], device=y_poz.device, dtype=torch.long)
    fark = y_poz[idx] - y_neg[idx]
    kayip = torch.clamp(marj - fark, min=0.0).mean()
    return kayip, fark.mean().detach(), (fark < marj).float().mean().detach()


def molecule_margin_loss(tahmin_afinite, targets, marj=1.0):
    """MOLECULE margin: within the same pocket group, y(binder) >= y(decoy) + margin.

    Difference from the pose margin: there the correct and perturbed POSES of the same molecule
    were compared; here the pocket's BINDER is compared against its DECOY MOLECULES. No extra
    forward pass is needed, since the affinity output of the main pass is enough (the decoys are
    already in the batch).

    Related work: the inequality constraint in PIGNet (a decoy must be at least one margin worse
    than the real structure). There is no absolute label: we do not know a decoy's true affinity.
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
