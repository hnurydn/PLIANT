#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_perturbation.py
=====================
Generates controlled, RMSD-graded ligand pose perturbations for CASF-2016
core-set complexes. Goal: measure how much affinity/contact prediction
quality degrades for PLANET models (particularly the Vector-Gating family)
as the input pose moves away from the crystal pose (see
makale/pose_perturbation_teknik_tasarim.md).

Input:  core.lmdb (chemutils.ComplexPocket records; key = pdb_id)
Output: one LMDB per RMSD bin (core_rmsd{R}.lmdb), plus a calibration QC
        report (CSV). The output LMDBs can be read by the existing eval
        scripts (eval_unified.py / PLANET_re-evaluate_*) WITH NO CODE CHANGES
        AT ALL — only the 3D conformer coordinates of ligand.mol change,
        everything else (the pro_lig_interaction label, pocket_residues, pK)
        stays NATIVE/ground-truth.

Key design decision: only the conformer of ligand.mol is perturbed;
pro_lig_interaction (the true contact map) and pK (the true affinity) are
left unchanged — because the question being asked is "given a wrong pose,
can the model still predict the true/native answer?"

Usage (try a small --limit first and check the calibration report):
    python pose_perturbation.py \
        --core-lmdb $PLANET_ROOT/datasets/pdbbind2020/lmdb_planet/core.lmdb \
        --out-dir   $PLANET_ROOT/datasets/pose_perturbation \
        --chemutils-path $PLANET_ROOT/Scripts \
        --rmsd-targets 0,1,2,3,4 \
        --mode rigid \
        --n-replicates 1 \
        --seed 42 \
        --limit 5          # for the first trial; remove for the full run

Required packages (all present in the pocketgen conda env, verified on TRUBA):
    rdkit, scipy, numpy, lmdb
"""

import argparse
import copy
import csv
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.spatial.transform import Rotation
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout




# ───────────────────────────── Geometri yardımcıları ─────────────────────────────

def get_conformer_array(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    """RDKit Mol'ün konformerini (N,3) float64 numpy array olarak döndürür."""
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float64)


def set_conformer_array(mol: Chem.Mol, coords: np.ndarray, conf_id: int = 0) -> None:
    """(N,3) koordinat array'ini RDKit Mol'ün konformerine yazar (in-place)."""
    conf = mol.GetConformer(conf_id)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].tolist())


def heavy_atom_mask(mol: Chem.Mol) -> np.ndarray:
    """Hidrojen olmayan atomların boolean maskesi (standart 'heavy-atom RMSD' pratiği)."""
    return np.array([atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()], dtype=bool)


def heavy_rmsd(coords_a: np.ndarray, coords_b: np.ndarray, mask: np.ndarray) -> float:
    """Hizalama YAPMADAN ham koordinat RMSD'si — poz-çerçevesindeki gerçek konum farkı
    ölçülmek isteniyor (Kabsch hizalaması perturbation'ı 'düzeltir', burada istenmiyor)."""
    a, b = coords_a[mask], coords_b[mask]
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ───────────────────────────── (a) Rigid-body perturbation ─────────────────────────────

def rigid_body_transform(
    coords: np.ndarray, axis: np.ndarray, angle_deg: float, trans_vec: np.ndarray
) -> np.ndarray:
    """Ağırlık merkezi etrafında rotasyon + sabit yönde translasyon uygular."""
    centroid = coords.mean(axis=0)
    rot = Rotation.from_rotvec(axis * np.deg2rad(angle_deg))
    rotated = rot.apply(coords - centroid) + centroid
    return rotated + trans_vec


def calibrate_rigid_body(
    mol: Chem.Mol,
    target_rmsd: float,
    rng: np.random.Generator,
    conf_id: int = 0,
    tol: float = 0.05,
    max_iter: int = 40,
    max_angle_deg: float = 120.0,
    max_trans_scale: float = 8.0,
) -> Tuple[np.ndarray, float, dict]:
    """Hedef heavy-atom RMSD'ye ulaşana kadar rotasyon açısı + translasyon büyüklüğünü
    tek bir skala parametresi 's' üzerinden binary search ile kalibre eder.

    Yön (rotasyon ekseni, translasyon yönü) SABİT rastgele seçilir; sadece büyüklük
    ('s') değişir -> f(s)=rmsd(s) yaklaşık monoton artan, binary search güvenli çalışır.
    """
    native = get_conformer_array(mol, conf_id)
    mask = heavy_atom_mask(mol)

    if target_rmsd <= 1e-9:
        return native.copy(), 0.0, {"converged": True, "scale": 0.0}

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    trans_dir = rng.normal(size=3)
    trans_dir /= np.linalg.norm(trans_dir)

    def f(s: float) -> float:
        trans_vec = trans_dir * (s * max_trans_scale)
        perturbed = rigid_body_transform(native, axis, s * max_angle_deg, trans_vec)
        return heavy_rmsd(perturbed, native, mask)

    lo, hi = 0.0, 1.0
    # Üst sınırın hedefi gerçekten aşıp aşmadığını doğrula; aşmıyorsa genişlet (küçük
    # ligandlarda/az atomda rotasyon RMSD'ye az katkı yapabilir, translasyon baskındır).
    f_hi = f(hi)
    expand_iter = 0
    while f_hi < target_rmsd and expand_iter < 5:
        max_trans_scale *= 1.5
        f_hi = f(hi)
        expand_iter += 1

    best_s, best_rmsd = hi, f_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        r = f(mid)
        if abs(r - target_rmsd) < abs(best_rmsd - target_rmsd):
            best_s, best_rmsd = mid, r
        if r < target_rmsd:
            lo = mid
        else:
            hi = mid
        if abs(r - target_rmsd) <= tol:
            best_s, best_rmsd = mid, r
            break

    trans_vec = trans_dir * (best_s * max_trans_scale)
    final_coords = rigid_body_transform(native, axis, best_s * max_angle_deg, trans_vec)
    converged = abs(best_rmsd - target_rmsd) <= tol
    return final_coords, best_rmsd, {
        "converged": converged, "scale": best_s,
        "axis": axis.tolist(), "trans_dir": trans_dir.tolist(),
    }


# ───────────────────────────── (b) Torsiyonel perturbation ─────────────────────────────

# RDKit'in standart döndürülebilir bağ SMARTS deseni (üçlü bağ ucu ve terminal atomlar hariç)
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def get_rotatable_dihedrals(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    """Her döndürülebilir bağ için (a1,a2,a3,a4) dihedral atom dörtlüsü döndürür.
    a2-a3 döndürülebilir bağın kendisi; a1/a4 her uçtan birer komşu atom."""
    dihedrals = []
    matches = mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS)
    for (a2, a3) in matches:
        atom2, atom3 = mol.GetAtomWithIdx(a2), mol.GetAtomWithIdx(a3)
        n1 = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != a3]
        n4 = [n.GetIdx() for n in atom3.GetNeighbors() if n.GetIdx() != a2]
        if not n1 or not n4:
            continue
        dihedrals.append((n1[0], a2, a3, n4[0]))
    return dihedrals


def calibrate_torsional(
    mol: Chem.Mol,
    target_rmsd: float,
    rng: np.random.Generator,
    conf_id: int = 0,
    tol: float = 0.05,
    max_iter: int = 40,
    max_delta_deg: float = 180.0,
    max_n_bonds: int = 6,
) -> Optional[Tuple[np.ndarray, float, dict]]:
    """Rastgele seçilmiş döndürülebilir bağların dihedral açılarını, tek bir skala
    parametresiyle hedef RMSD'ye kalibre ederek bozar. Döndürülebilir bağ yoksa None
    döner (küçük/rijit ligandlarda torsiyonel perturbation uygulanamaz — bu durum
    kalibrasyon QC raporunda ayrıca işaretlenmeli)."""
    native = get_conformer_array(mol, conf_id)
    mask = heavy_atom_mask(mol)

    dihedrals = get_rotatable_dihedrals(mol)
    if not dihedrals:
        return None
    rng.shuffle(dihedrals)
    dihedrals = dihedrals[:max_n_bonds]
    signs = rng.choice([-1.0, 1.0], size=len(dihedrals))

    if target_rmsd <= 1e-9:
        return native.copy(), 0.0, {"converged": True, "scale": 0.0, "n_bonds": len(dihedrals)}

    work_mol = Chem.Mol(mol)
    work_conf = work_mol.GetConformer(conf_id)

    def f(s: float) -> float:
        set_conformer_array(work_mol, native, conf_id)  # resetle
        for (a1, a2, a3, a4), sign in zip(dihedrals, signs):
            try:
                current = rdMolTransforms.GetDihedralDeg(work_conf, a1, a2, a3, a4)
                rdMolTransforms.SetDihedralDeg(
                    work_conf, a1, a2, a3, a4, current + sign * s * max_delta_deg
                )
            except Exception:
                continue  # halka içi/geçersiz dihedral -> atla
        perturbed = get_conformer_array(work_mol, conf_id)
        return heavy_rmsd(perturbed, native, mask)

    lo, hi = 0.0, 1.0
    f_hi = f(hi)
    if f_hi < target_rmsd:
        # Bu ligand için mevcut döndürülebilir bağlarla hedef RMSD'ye ulaşılamıyor.
        final_coords = get_conformer_array(work_mol, conf_id)
        return final_coords, f_hi, {
            "converged": False, "scale": hi, "n_bonds": len(dihedrals),
            "reason": "target_unreachable_with_available_torsions",
        }

    best_s, best_rmsd = hi, f_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        r = f(mid)
        if abs(r - target_rmsd) < abs(best_rmsd - target_rmsd):
            best_s, best_rmsd = mid, r
        if r < target_rmsd:
            lo = mid
        else:
            hi = mid
        if abs(r - target_rmsd) <= tol:
            best_s, best_rmsd = mid, r
            break

    final_rmsd = f(best_s)
    final_coords = get_conformer_array(work_mol, conf_id)
    converged = abs(final_rmsd - target_rmsd) <= tol
    return final_coords, final_rmsd, {
        "converged": converged, "scale": best_s, "n_bonds": len(dihedrals),
    }


# ───────────────────────────── LMDB okuma/yazma ─────────────────────────────

@dataclass
class QCRow:
    pdb_id: str
    rmsd_bin: float
    mode: str
    replicate: int
    achieved_rmsd: float
    n_atoms: int
    n_rotatable_bonds: int
    converged: bool
    note: str = ""


def load_core_lmdb_keys(env) -> List[bytes]:
    with env.begin(write=False) as txn:
        return [k for k, _ in txn.cursor()]


def load_pocket(env, key: bytes):
    with env.begin(write=False) as txn:
        raw = txn.get(key)
    return pickle.loads(bytes(raw))


def write_lmdb(records: dict, out_path: str, map_size_gb: int = 4):
    """records: {key_bytes: pickled_bytes}"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = lmdb.open(out_path, map_size=map_size_gb * 1024 ** 3)
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k, v)
    env.close()


# ───────────────────────────── Ana işlem ─────────────────────────────

def perturb_pocket_ligand(
    pocket, target_rmsd: float, mode: str, rng: np.random.Generator
) -> Tuple[object, float, dict]:
    """pocket.ligand.mol'ün YENİ bir kopyasını üretip perturbe eder; orijinal pocket
    nesnesini (ve onun native mol'ünü) DEĞİŞTİRMEZ — çağıran taraf her bin için
    aynı native pocket'ı tekrar tekrar güvenle kullanabilir."""
    native_mol = pocket.ligand.mol
    n_atoms = native_mol.GetNumAtoms()
    n_rot = len(get_rotatable_dihedrals(native_mol))

    if mode == "rigid":
        coords, achieved, meta = calibrate_rigid_body(native_mol, target_rmsd, rng)
    elif mode == "torsional":
        result = calibrate_torsional(native_mol, target_rmsd, rng)
        if result is None:
            # Döndürülebilir bağ yok -> rigid-body'ye düş (QC raporunda not düşülür)
            coords, achieved, meta = calibrate_rigid_body(native_mol, target_rmsd, rng)
            meta["note"] = "no_rotatable_bonds_fallback_to_rigid"
        else:
            coords, achieved, meta = result
    else:
        raise ValueError(f"Bilinmeyen mode: {mode}")

    new_mol = Chem.Mol(native_mol)  # atom/bağ topolojisi + konformer şablonu kopyalanır
    set_conformer_array(new_mol, coords)

    # chemutils.Mol sarmalayıcısı — pocket.ligand ile aynı sınıf, aynı modülden
    MolWrapper = type(pocket.ligand)
    new_ligand = MolWrapper(new_mol)

    new_pocket = copy.copy(pocket)  # sığ kopya: pocket_residues/pro_lig_interaction PAYLAŞILIR (native kalır)
    new_pocket.ligand = new_ligand
    # QC/iz sürme için ek metadata (eval script'i görmezden gelebilir, dict/attr olarak eklenir)
    new_pocket.pose_perturbation_meta = {
        "target_rmsd": target_rmsd, "achieved_rmsd": achieved,
        "mode": mode, "n_atoms": n_atoms, "n_rotatable_bonds": n_rot, **meta,
    }
    return new_pocket, achieved, {"n_atoms": n_atoms, "n_rotatable_bonds": n_rot, **meta}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core-lmdb", required=True, help="Girdi core.lmdb yolu")
    ap.add_argument("--out-dir", required=True, help="Çıktı LMDB'lerinin ve QC raporunun yazılacağı dizin")
    ap.add_argument("--chemutils-path", required=True,
                     help="chemutils.py'nin bulunduğu dizin (Scripts/), sys.path'e eklenir")
    ap.add_argument("--rmsd-targets", default="0,1,2,3,4",
                     help="Virgülle ayrılmış hedef RMSD bin'leri (Å)")
    ap.add_argument("--mode", choices=["rigid", "torsional", "both"], default="rigid")
    ap.add_argument("--n-replicates", type=int, default=1,
                     help="Her (kompleks, bin) için kaç bağımsız rastgele perturbation üretilecek")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                     help="Test için ilk N kompleksle sınırla (tam koşuda kaldırın)")
    ap.add_argument("--tol", type=float, default=0.05, help="RMSD kalibrasyon toleransı (Å)")
    args = ap.parse_args()

    sys.path.insert(0, args.chemutils_path)
    import chemutils  # noqa: F401  (pickle.loads'un modülü bulabilmesi için import edilmesi yeterli)

    rmsd_targets = [float(x) for x in args.rmsd_targets.split(",")]
    modes = ["rigid", "torsional"] if args.mode == "both" else [args.mode]

    in_env = lmdb.open(args.core_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    keys = load_core_lmdb_keys(in_env)
    if args.limit:
        keys = keys[: args.limit]
    print(f"[pose_perturbation] {len(keys)} kompleks işlenecek, "
          f"RMSD hedefleri={rmsd_targets}, mod(lar)={modes}, replikasyon={args.n_replicates}")

    qc_rows: List[QCRow] = []
    # bin/mode/replikasyon başına ayrı LMDB dosyası -> eval script'i değişmeden okuyabilir
    out_buffers = {}  # (rmsd_bin, mode, rep) -> {key: pickled_bytes}

    for idx, key in enumerate(keys):
        pdb_id = key.decode()
        pocket = load_pocket(in_env, key)

        for mode in modes:
            for target_rmsd in rmsd_targets:
                for rep in range(args.n_replicates):
                    seed = args.seed + hash((pdb_id, mode, target_rmsd, rep)) % (2**31)
                    rng = np.random.default_rng(seed)
                    try:
                        new_pocket, achieved, meta = perturb_pocket_ligand(
                            pocket, target_rmsd, mode, rng
                        )
                    except Exception as e:
                        print(f"  [!] {pdb_id} rmsd={target_rmsd} mode={mode} rep={rep} HATA: {e}")
                        qc_rows.append(QCRow(pdb_id, target_rmsd, mode, rep, float("nan"),
                                              pocket.ligand.mol.GetNumAtoms(), -1, False, str(e)))
                        continue

                    out_key = (target_rmsd, mode, rep)
                    out_buffers.setdefault(out_key, {})[key] = pickle.dumps(new_pocket)

                    qc_rows.append(QCRow(
                        pdb_id, target_rmsd, mode, rep, achieved,
                        meta["n_atoms"], meta["n_rotatable_bonds"], meta.get("converged", False),
                        meta.get("note", meta.get("reason", "")),
                    ))

        if (idx + 1) % 25 == 0 or idx == len(keys) - 1:
            print(f"  ... {idx + 1}/{len(keys)} kompleks işlendi")

    in_env.close()

    # ─ Çıktıları yaz ─
    os.makedirs(args.out_dir, exist_ok=True)
    for (target_rmsd, mode, rep), records in out_buffers.items():
        fname = f"core_rmsd{target_rmsd:g}_{mode}"
        if args.n_replicates > 1:
            fname += f"_rep{rep}"
        fname += ".lmdb"
        out_path = os.path.join(args.out_dir, fname)
        write_lmdb(records, out_path)
        print(f"[pose_perturbation] yazıldı: {out_path}  ({len(records)} kompleks)")

    # ProLigDataset her LMDB icin yaninda ayni-isimli _meta.json bekliyor (pdb_id/pK/
    # n_decoys listesi) -- icerik perturbation'dan etkilenmez (pK/decoy sayisi degismedi,
    # sadece ligand geometrisi degisti), bu yuzden kaynak core_meta.json'u dogrudan kopyala.
    src_meta = args.core_lmdb.rstrip("/\\")
    if src_meta.endswith(".lmdb"):
        src_meta = src_meta[:-5]
    src_meta += "_meta.json"
    if os.path.exists(src_meta):
        import shutil
        for (target_rmsd, mode, rep) in out_buffers:
            fname = f"core_rmsd{target_rmsd:g}_{mode}"
            if args.n_replicates > 1:
                fname += f"_rep{rep}"
            meta_out = os.path.join(args.out_dir, fname + "_meta.json")
            shutil.copyfile(src_meta, meta_out)
        print(f"[pose_perturbation] meta.json dosyaları kopyalandı (kaynak: {src_meta})")
    else:
        print(f"[pose_perturbation] UYARI: kaynak meta.json bulunamadı ({src_meta}) — "
              f"ProLigDataset bu LMDB'leri okuyamayabilir, elle kopyalayın.")

    qc_path = os.path.join(args.out_dir, "calibration_report.csv")
    with open(qc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "rmsd_bin_target", "mode", "replicate", "achieved_rmsd",
                          "n_atoms", "n_rotatable_bonds", "converged", "note"])
        for row in qc_rows:
            writer.writerow([row.pdb_id, row.rmsd_bin, row.mode, row.replicate,
                              f"{row.achieved_rmsd:.4f}" if not np.isnan(row.achieved_rmsd) else "NaN",
                              row.n_atoms, row.n_rotatable_bonds, row.converged, row.note])
    print(f"[pose_perturbation] kalibrasyon QC raporu: {qc_path}")

    # Hızlı özet: hedeften >tol sapan / converge olmayan oran
    n_total = len(qc_rows)
    n_bad = sum(1 for r in qc_rows if not r.converged and r.rmsd_bin > 0)
    n_nonzero = sum(1 for r in qc_rows if r.rmsd_bin > 0)
    if n_nonzero:
        print(f"[pose_perturbation] Kalibrasyon özeti: {n_bad}/{n_nonzero} "
              f"(bin>0) tolerans ({args.tol} Å) dışında kaldı.")


if __name__ == "__main__":
    main()
