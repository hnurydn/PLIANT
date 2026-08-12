"""
prepare_benchmark_proteins.py
==============================
DUDE ve LIT-PCBA protein PDB dosyalarını PDBFixer ile hazırlar.
Hazırlanan dosyalar orijinal ile aynı klasöre _prepared.pdb olarak kaydedilir.
JSON haritası güncellenerek protein_pdb → protein_prepared_pdb eklenir.

Kullanım:
    python prepare_benchmark_proteins.py \
        --benchmark dude \
        --json_path /arf/.../dude_paths_prepared_arf.json \
        --njobs 20

    python prepare_benchmark_proteins.py \
        --benchmark litpcba \
        --json_path /arf/.../lit_pcba_paths_prepared_arf.json \
        --njobs 20
"""

import os, sys, json, argparse
from multiprocessing import Pool


def prepare_single(pdb_path):
    """Tek bir PDB dosyasını PDBFixer ile hazırlar."""
    out_path = pdb_path.replace('.pdb', '_prepared.pdb')

    if os.path.exists(out_path):
        print(f'[ATLANDI] {os.path.basename(pdb_path)}', flush=True)
        return (pdb_path, out_path, 'exists')

    try:
        import pdbfixer
        from simtk.openmm.app import PDBFile

        fixer = pdbfixer.PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.removeHeterogens(False)
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        with open(out_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        print(f'[BAŞARILI] {os.path.basename(pdb_path)}', flush=True)
        return (pdb_path, out_path, None)

    except Exception as e:
        print(f'[HATA] {os.path.basename(pdb_path)}: {e}', flush=True)
        return (pdb_path, None, str(e))


def collect_dude_paths(json_path):
    """DUDE JSON'undan protein_pdb yollarını toplar."""
    with open(json_path) as f:
        data = json.load(f)

    paths = []
    for target, records in data.items():
        if records:
            pdb = records[0].get('protein_pdb', '')
            if pdb and os.path.exists(pdb):
                paths.append(pdb)
            else:
                print(f'[UYARI] DUDE protein bulunamadı: {target} → {pdb}')
    return paths, data


def collect_litpcba_paths(json_path):
    """LIT-PCBA JSON'undan protein_file (.pdb) yollarını toplar."""
    with open(json_path) as f:
        data = json.load(f)

    paths = []
    for target, templates in data.items():
        for tmpl in templates:
            # .pdb versiyonunu bul
            mol2_path = tmpl.get('protein_file', '')
            pdb_path  = mol2_path.replace('.mol2', '.pdb')
            if os.path.exists(pdb_path):
                paths.append(pdb_path)
            elif os.path.exists(mol2_path):
                print(f'[UYARI] Sadece mol2 var, pdb yok: {pdb_path}')
            else:
                print(f'[UYARI] Protein bulunamadı: {target}/{tmpl.get("name","?")}')
    return paths, data


def update_dude_json(data, results, json_path):
    """DUDE JSON'una protein_prepared_pdb ekler."""
    path_map = {orig: prep for orig, prep, err in results if prep}

    for target, records in data.items():
        for record in records:
            orig = record.get('protein_pdb', '')
            if orig in path_map:
                record['protein_prepared_pdb'] = path_map[orig]

    out_path = json_path.replace('.json', '_pdbfixed.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n✅ Güncellenmiş JSON: {out_path}')
    return out_path


def update_litpcba_json(data, results, json_path):
    """LIT-PCBA JSON'una protein_prepared_pdb ekler."""
    path_map = {}
    for orig, prep, err in results:
        if prep:
            mol2 = orig.replace('.pdb', '.mol2')
            path_map[mol2] = prep
            path_map[orig] = prep

    for target, templates in data.items():
        for tmpl in templates:
            mol2 = tmpl.get('protein_file', '')
            pdb  = mol2.replace('.mol2', '.pdb')
            if mol2 in path_map:
                tmpl['protein_prepared_pdb'] = path_map[mol2]
            elif pdb in path_map:
                tmpl['protein_prepared_pdb'] = path_map[pdb]

    out_path = json_path.replace('.json', '_pdbfixed.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n✅ Güncellenmiş JSON: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, choices=['dude', 'litpcba'])
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--njobs',     type=int, default=20)
    args = parser.parse_args()

    print(f'Benchmark: {args.benchmark.upper()}')
    print(f'JSON: {args.json_path}')
    print(f'CPU: {args.njobs}')

    if args.benchmark == 'dude':
        paths, data = collect_dude_paths(args.json_path)
    else:
        paths, data = collect_litpcba_paths(args.json_path)

    print(f'\nToplam protein: {len(paths)}')

    # Zaten hazırlanmış olanları filtrele
    to_process = [p for p in paths if not os.path.exists(p.replace('.pdb', '_prepared.pdb'))]
    already    = len(paths) - len(to_process)
    print(f'Zaten hazır: {already}, İşlenecek: {len(to_process)}')

    if to_process:
        with Pool(args.njobs) as pool:
            new_results = pool.map(prepare_single, to_process)
    else:
        new_results = []

    # Zaten hazır olanları da results'a ekle
    already_results = [(p, p.replace('.pdb', '_prepared.pdb'), 'exists')
                       for p in paths if os.path.exists(p.replace('.pdb', '_prepared.pdb'))]
    all_results = already_results + new_results

    # Özet
    success = sum(1 for _, p, _ in all_results if p)
    errors  = sum(1 for _, p, e in all_results if not p)
    print(f'\n{"="*50}')
    print(f'Başarılı : {success}')
    print(f'Hata     : {errors}')
    print('='*50)

    # JSON güncelle
    if args.benchmark == 'dude':
        update_dude_json(data, all_results, args.json_path)
    else:
        update_litpcba_json(data, all_results, args.json_path)


if __name__ == '__main__':
    main()