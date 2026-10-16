#!/usr/bin/env python3
"""Full placement pass (19 Sep 2026) - a shared work queue plus local Vina optimisation.

For every decoy: geometric placement (radius-based, v3) -> LOCAL Vina optimisation (no search) ->
the pose is written back BY MATCHING COORDINATES (atom order cannot be trusted; see the 18 Sep test).
The unit of work is a (pocket, decoy shard) pair. Shards are taken from a shared queue under flock,
so there is no queue contention and more work can be appended later.
Output: yerlesim/son/<split>_<key>_<start>.npz (plus energies and clash values)
Usage: SPLIT=train NPROC=56 python -u place_decoys.py
"""
import sys, os, time, json, pickle, subprocess, tempfile, fcntl
import numpy as np
sys.path.insert(0, "/arf/scratch/hnaydin/PLANET_FIXED_20260917/pipeline")
sys.path.insert(0, "/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte")
import lmdb, chemutils_esm2_vector
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')
from vina import Vina
from inpocket_placement import place_in_pocket, vdw_radius
from multiprocessing import Pool

OB = "/arf/home/hnaydin/miniconda3/envs/pocketgen/bin/obabel"
P = "/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020"
L = os.environ.get("LMDB_DIR", f"{P}/lmdb_planet_v2")   # copied to /tmp at the start of the job
KOK = "/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/yerlesim"
OUT = f"{KOK}/son"
SPLIT = os.environ.get("SPLIT", "core")
DILIM = int(os.environ.get("DILIM", "150"))
MAXDEC = int(os.environ.get("MAXDEC", "0"))   # 0 = unlimited; e.g. 100 = first 100 decoys per pocket
NPROC = int(os.environ.get("NPROC", "20"))
ZAMANLA = os.environ.get("ZAMANLA", "0") == "1"   # 1 = log per-stage timings for every shard
AD_OK = {"C","N","O","S","P","H","F","CL","BR","I","ZN","FE","MG","CA","MN","CU","NA","K","HG","CD","NI","CO","SR"}
KUYRUK = f"{KOK}/kuyruk_{SPLIT}" + (f"_maks{MAXDEC}" if MAXDEC else "") + ".txt"
IMLEC = f"{KOK}/imlec_{SPLIT}" + (f"_maks{MAXDEC}" if MAXDEC else "") + ".txt"
# [20 Sep] use a prepared queue file: KUYRUK_EK=kalan -> kuyruk_{split}_kalan.txt
_EK = os.environ.get("KUYRUK_EK", "")
if _EK:
    KUYRUK = f"{KOK}/kuyruk_{SPLIT}_{_EK}.txt"
    IMLEC  = f"{KOK}/imlec_{SPLIT}_{_EK}.txt"
os.makedirs(OUT, exist_ok=True)


def kuyruk_kur():
    """Build the (pocket, start, end) work list once."""
    if os.path.exists(KUYRUK) and os.path.getsize(KUYRUK) > 0: return
    meta = json.load(open(f"{L}/{SPLIT}_meta.json"))
    sat = []
    for r in meta:
        n = r.get("n_decoys", 0)
        if MAXDEC: n = min(n, MAXDEC)
        if n <= 0: continue
        for s in range(0, n, DILIM):
            sat.append(f"{r['key']}\t{s}\t{min(s + DILIM, n)}")
    tmp = KUYRUK + ".tmp"
    open(tmp, "w").write("\n".join(sat) + "\n")
    os.replace(tmp, KUYRUK)
    print(f"[kuyruk] {len(sat)} iş birimi ({SPLIT})", flush=True)


def sirada_al():
    """Return the next line from the queue, taking a lock (None = queue exhausted)."""
    fd = os.open(IMLEC, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, 0)
        ham = os.read(fd, 64).decode().strip()
        i = int(ham) if ham else 0
        os.lseek(fd, 0, 0); os.write(fd, f"{i+1}\n".encode()); os.ftruncate(fd, len(f"{i+1}\n"))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    with open(KUYRUK) as f:
        for j, l in enumerate(f):
            if j == i: return i, l.strip()
    return i, None


def find_dir(k):
    for y in ("2011-2019", "2001-2010", "1981-2000"):
        d = f"{P}/{y}/{k}"
        if os.path.exists(f"{d}/{k}_pocket.pdb"): return d


def pdbqt_koord(path):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in open(path) if l.startswith(("ATOM", "HETATM"))])


ESKI_SAHIP = 1800   # seconds: a claim file older than this is treated as stale, so unfinished work is taken over


def sahiplen(hedef):
    """Claim a shard atomically (O_EXCL); take over a stale claim."""
    cl = hedef + ".claim"
    try:
        fd = os.open(cl, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(cl) > ESKI_SAHIP:
                os.utime(cl, None)      # stale -> take over
                return True
        except FileNotFoundError:
            pass
        return False


def eksik_dilimler():
    """Shards with no output, including those cut short by the time limit."""
    out = []
    for l in open(KUYRUK):
        l = l.strip()
        if not l: continue
        k, a, _ = l.split("\t")
        if not os.path.exists(f"{OUT}/{SPLIT}_{k}_{a}.npz"): out.append(l)
    return out


def isci(_):
    env = None
    islenen = 0
    tur, eksikler = 1, []
    while True:
        if tur == 1:
            idx, sat = sirada_al()
            if sat is None:
                tur = 2
                eksikler = eksik_dilimler()
                print(f"  [2. tur] {len(eksikler)} eksik dilim", flush=True)
                continue
        else:
            if not eksikler: break
            sat = eksikler.pop()
        key, a, b = sat.split("\t"); a, b = int(a), int(b)
        hedef = f"{OUT}/{SPLIT}_{key}_{a}.npz"
        if os.path.exists(hedef): continue
        if not sahiplen(hedef): continue
        tmp = tempfile.mkdtemp(prefix=f"y4_{os.getpid()}_")
        ts0 = time.time(); t_yer = t_vina = t_map = 0.0
        try:
            d = find_dir(key)
            if d is None: continue
            if env is None:
                env = lmdb.open(f"{L}/{SPLIT}.lmdb", readonly=True, lock=False, readahead=False, max_readers=1024)
            with env.begin() as txn: raw = txn.get(key.encode())
            if raw is None: continue
            obj = pickle.loads(bytes(raw))
            decs = (obj.decoys or [])[a:b]
            if not decs: continue
            # pocket: coordinates plus element symbols (from the PDB; the LMDB has no elements)
            cep, cep_el = [], []
            with open(f"{tmp}/r.pdb", "w") as w:
                for l in open(f"{d}/{key}_pocket.pdb"):
                    if l.startswith(("ATOM", "HETATM")):
                        e = (l[76:78].strip() or l[12:16].strip()[:2]).upper()
                        if e not in AD_OK and e[:1] not in AD_OK: continue
                        cep.append([float(l[30:38]), float(l[38:46]), float(l[46:54])]); cep_el.append(e if e in AD_OK else e[:1])
                    w.write(l)
            cep = np.array(cep); r_cep = vdw_radius(cep_el)
            lig = obj.ligand.mol; lp = lig.GetConformer().GetPositions(); ref = lp.mean(0)
            subprocess.run([OB, f"{tmp}/r.pdb", "-xr", "-h", "-O", f"{tmp}/r.pdbqt"], capture_output=True)
            v, vina_ok = None, False
            tm = time.time()
            try:
                v = Vina(sf_name="vina", cpu=1, verbosity=0); v.set_receptor(f"{tmp}/r.pdbqt")
                size = np.minimum(np.maximum((lp.max(0) - lp.min(0)) + 12.0, 20.0), 30.0)
                v.compute_vina_maps(center=list(ref), box_size=list(size)); vina_ok = True
            except Exception:
                v = None
            t_map = time.time() - tm
            rng = np.random.default_rng((abs(hash(key)) + a) % 2**31)
            xs, e_once, e_son, cakl, nat = [], [], [], [], []
            for dm in decs:
                m = Chem.Mol(dm.mol); x0 = m.GetConformer().GetPositions()
                sym = [at.GetSymbol() for at in m.GetAtoms()]
                agir = np.array([at.GetAtomicNum() > 1 for at in m.GetAtoms()])
                ty = time.time()
                try:
                    x, E, cak, tem, sap = place_in_pocket(x0, agir, vdw_radius(sym)[agir], cep, r_cep, ref, rng,
                                                       F=0.85, ALFA=0.05, n_aday=int(os.environ.get("NADAY", "64")), n_baslangic=1, n_adim=25)
                except Exception:
                    continue
                t_yer += time.time() - ty
                e1 = e2 = np.nan
                tv = time.time()
                if vina_ok:
                    try:
                        c = m.GetConformer()
                        for i, p in enumerate(x): c.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
                        w = Chem.SDWriter(f"{tmp}/l.sdf"); w.write(m); w.close()
                        if os.path.exists(f"{tmp}/l.pdbqt"): os.remove(f"{tmp}/l.pdbqt")
                        subprocess.run([OB, f"{tmp}/l.sdf", "-O", f"{tmp}/l.pdbqt", "-xh"], capture_output=True)
                        if os.path.exists(f"{tmp}/l.pdbqt"):
                            pc = pdbqt_koord(f"{tmp}/l.pdbqt")
                            dm2 = np.sqrt(((pc[:, None, :] - x[None, :, :]) ** 2).sum(-1))
                            esl = dm2.argmin(1)
                            if dm2.min(1).max() < 0.05 and len(set(esl.tolist())) == len(esl):
                                v.set_ligand_from_string(open(f"{tmp}/l.pdbqt").read())
                                e1 = float(v.score()[0]); e2 = float(v.optimize()[0])
                                v.write_pose(f"{tmp}/p.pdbqt", overwrite=True)
                                pp = pdbqt_koord(f"{tmp}/p.pdbqt")
                                if len(pp) == len(esl):
                                    xn = x.copy(); xn[esl] = pp
                                    bag = [np.linalg.norm(xn[bb.GetBeginAtomIdx()] - xn[bb.GetEndAtomIdx()]) for bb in m.GetBonds()]
                                    if not bag or (min(bag) > 0.8 and max(bag) < 2.4): x = xn
                                os.remove(f"{tmp}/p.pdbqt")
                    except Exception:
                        pass
                t_vina += time.time() - tv
                xs.append(x.astype(np.float32)); e_once.append(e1); e_son.append(e2); cakl.append(cak); nat.append(len(x0))
            if not xs: continue
            nmax = max(nat)
            arr = np.full((len(xs), nmax, 3), np.nan, dtype=np.float32)
            for i, x in enumerate(xs): arr[i, :len(x)] = x
            np.savez_compressed(hedef + ".tmp.npz", xyz=arr, nat=np.array(nat, np.int32),
                                e_once=np.array(e_once, np.float32), e_son=np.array(e_son, np.float32),
                                cakisma=np.array(cakl, np.float32), bas=np.int32(a), vina_ok=np.int8(vina_ok))
            os.replace(hedef + ".tmp.npz", hedef)
            islenen += len(xs)
            if ZAMANLA:
                tt = time.time() - ts0
                print(f"  [sure] {key} {a} n={len(xs)} tot={tt:.0f}s yer={t_yer:.0f} vina={t_vina:.0f} "
                      f"map={t_map:.0f} diger={tt-t_yer-t_vina-t_map:.0f} | {tt/max(1,len(xs)):.2f} s/decoy", flush=True)
            try: os.remove(hedef + ".claim")
            except OSError: pass
        except Exception as ex:
            print(f"  [hata] {key} {a}: {type(ex).__name__}: {str(ex)[:80]}", flush=True)
        finally:
            subprocess.run(["rm", "-rf", tmp], capture_output=True)
    return islenen


if __name__ == "__main__":
    kuyruk_kur()
    n_is = sum(1 for _ in open(KUYRUK))
    t0 = time.time()
    print(f"[{SPLIT}] kuyruk {n_is} iş birimi | NPROC={NPROC} | dilim={DILIM}", flush=True)
    with Pool(NPROC) as pool:
        toplam = sum(pool.map(isci, range(NPROC)))
    bitti = len(os.listdir(OUT))
    print(f"BITTI: bu iş {toplam} decoy işledi | {(time.time()-t0)/60:.1f} dk | dizinde {bitti} parça", flush=True)
