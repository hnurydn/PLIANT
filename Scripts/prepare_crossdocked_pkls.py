"""
prepare_crossdocked_pkls.py
============================
CrossDocked 2020 verisi için ComplexPocket PKL dosyaları oluşturur.

PDBbind'den farkı:
  - Her pocket için birden fazla binder var (cross-docked ligandlar)
  - Decoy yok — tüm ligandlar complex_label=1 (binder)
  - pK değerleri JSON'dan gelir (Vina tabanlı)

Çıktı yapısı:
  output_dir/
    {pocket_name}/
      {pdb_id}_{lig_code}_binder.pkl   ← her binder için ayrı pkl

Kullanım:
    python prepare_crossdocked_pkls.py \
        --json_path /arf/.../crossdocked_new_records.json \
        --extracted_dir /arf/.../crossdocked_extracted \
        --output_dir /arf/.../crossdocked_pkls \
        --njobs 8
"""

import os, sys, json, pickle, argparse, glob
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chemutils_esm2 import ComplexPocket   # ESM2 versiyonu
except ImportError as e:
    print(f"KRİTİK HATA: {e}", file=sys.stderr)
    sys.exit(1)


# ── SDF eşleştirme ────────────────────────────────────────────────────────────
def find_sdf(folder_path, lig_file):
    """
    gninatypes dosya adından SDF dosyasını bulur.
    Örnek:
      5oeh_A_rec_4dhs_y03_lig_tt_min_0.gninatypes → 4dhs_y03_uff.sdf
      3t0m_A_rec_4dhr_y04_lig_tt_docked_1.gninatypes → 4dhr_y04_uff2.sdf
    """
    base = lig_file.replace('.gninatypes', '')

    if 'tt_min' in base:
        core     = base.split('_tt_min')[0].replace('_lig', '')
        lig_code = core.split('_rec_')[1] if '_rec_' in core else None
        candidates = [f'{lig_code}_uff.sdf', f'{lig_code}_uff2.sdf'] if lig_code else []
    else:
        core     = base.split('_tt_docked')[0].replace('_lig', '')
        lig_code = core.split('_rec_')[1] if '_rec_' in core else None
        candidates = [f'{lig_code}_uff2.sdf', f'{lig_code}_uff.sdf'] if lig_code else []

    for cand in candidates:
        path = os.path.join(folder_path, cand)
        if os.path.exists(path):
            return path, lig_code

    return None, lig_code


# ── Receptor PDB eşleştirme ───────────────────────────────────────────────────
def find_receptor_pdb(folder_path, pdb_id):
    pdb_file = os.path.join(folder_path, f'{pdb_id}_A_rec.pdb')
    if os.path.exists(pdb_file):
        return pdb_file
    matches = glob.glob(os.path.join(folder_path, f'{pdb_id}_*.pdb'))
    return matches[0] if matches else None


# ── Tek kayıt işleme ──────────────────────────────────────────────────────────
def process_record(record, extracted_dir, output_dir, overwrite=False):
    """Tek bir JSON kaydından ComplexPocket PKL oluşturur."""
    pdb_id   = record['pdb_id']
    pk       = record.get('pk', 0.0)
    folder   = record['ligand'].split('/')[0]
    lig_file = record['ligand'].split('/')[-1]

    folder_path = os.path.join(extracted_dir, folder)
    pocket_out  = os.path.join(output_dir, folder)
    os.makedirs(pocket_out, exist_ok=True)

    # SDF bul
    sdf_path, lig_code = find_sdf(folder_path, lig_file)
    if sdf_path is None or lig_code is None:
        return f"⚠️  SDF bulunamadı: {lig_file}"

    # Receptor PDB bul
    rec_pdb = find_receptor_pdb(folder_path, pdb_id)
    if rec_pdb is None:
        return f"⚠️  Receptor PDB bulunamadı: {pdb_id} in {folder}"

    # PKL çıktı yolu
    pkl_name = f"{pdb_id}_{lig_code}_binder.pkl"
    pkl_path = os.path.join(pocket_out, pkl_name)

    if os.path.exists(pkl_path) and not overwrite:
        return f"✓ ATLA: {pkl_name}"

    # ComplexPocket oluştur
    try:
        pocket = ComplexPocket(rec_pdb, sdf_path, pk, decoy_sdf="")
        with open(pkl_path, 'wb') as f:
            pickle.dump(pocket, f, pickle.HIGHEST_PROTOCOL)
        return f"✅ {pkl_name}"
    except Exception as e:
        return f"❌ HATA {pkl_name}: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='CrossDocked için ComplexPocket PKL oluşturur')
    parser.add_argument('--json_path',     required=True)
    parser.add_argument('--extracted_dir', required=True)
    parser.add_argument('--output_dir',    required=True)
    parser.add_argument('--njobs',         type=int, default=8)
    parser.add_argument('--limit',         type=int, default=None,
                        help='Test için ilk N kaydı işle')
    parser.add_argument('--overwrite',     action='store_true',
                        help='Mevcut PKL dosyalarının üzerine yaz')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("CrossDocked ComplexPocket PKL Hazırlama (Best-Pose Seçimi)")
    print("=" * 60)

    # JSON yükle ve best-pose seç
    # Her pocket-ligand ikilisi için: tt_min varsa onu al, yoksa tt_docked
    raw_records = json.load(open(args.json_path))

    best = {}
    for r in raw_records:
        folder   = r['ligand'].split('/')[0]
        lig_file = r['ligand'].split('/')[-1]
        base     = lig_file.replace('.gninatypes', '')
        if 'tt_min' in base:
            core = base.split('_tt_min')[0].replace('_lig', '')
        else:
            core = base.split('_tt_docked')[0].replace('_lig', '')
        lig_code = core.split('_rec_')[1] if '_rec_' in core else None
        if not lig_code:
            continue
        key  = f"{folder}__{r['pdb_id']}_{lig_code}"
        pose = 'min' if 'tt_min' in lig_file else 'docked'
        if key not in best or pose == 'min':
            best[key] = r

    records = list(best.values())
    if args.limit:
        records = records[:args.limit]

    tt_min_count    = sum(1 for r in records if 'tt_min'    in r['ligand'])
    tt_docked_count = sum(1 for r in records if 'tt_docked' in r['ligand'])
    print(f"Ham kayıt        : {len(raw_records)}")
    print(f"Best-pose seçilen: {len(records)}")
    print(f"  tt_min (crystal): {tt_min_count}")
    print(f"  tt_docked       : {tt_docked_count}")

    # Paralel işle
    worker = partial(process_record,
                     extracted_dir=args.extracted_dir,
                     output_dir=args.output_dir,
                     overwrite=args.overwrite)

    errors = skipped = success = 0

    with Pool(args.njobs) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker, records),
            total=len(records),
            desc="PKL oluşturuluyor"
        ))

    for r in results:
        if r.startswith("✅"):
            success += 1
        elif r.startswith("✓"):
            skipped += 1
        else:
            errors += 1

    print(f"\n{'='*50}")
    print(f"✅ Başarılı : {success}")
    print(f"✓  Atlandı  : {skipped}")
    print(f"❌ Hata     : {errors}")
    print(f"Toplam      : {len(results)}")
    print(f"\nPKL'ler kaydedildi: {args.output_dir}")


if __name__ == '__main__':
    main()