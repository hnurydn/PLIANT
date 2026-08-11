"""
Reconstruct per-checkpoint test-set metric trajectories for all 7 model
variants directly from the WandB training logs (project `PLANET_TRAINING`),
without re-running any GPU evaluation.

Each model's training was split across multiple WandB runs (checkpoint/resume
boundaries after Slurm 3-day time limits), so histories are merged by `_step`
across the listed run IDs before being written out as a single tidy CSV.

Requires: `wandb` (`pip install wandb`) and read access to the
`h-nurydn-hacettepe-university/PLANET_TRAINING` WandB project.

Usage:
    python fetch_checkpoint_curves_from_wandb.py

Output:
    checkpoint_curves_from_wandb.csv
    columns: model, step, Test_Affinity_Pearson, Test_Affinity_MAE,
             Test_Contact_AUROC, Test_Contact_AUPRC
"""

import csv

import wandb

ENTITY = "h-nurydn-hacettepe-university"
PROJECT = "PLANET_TRAINING"

# model name -> WandB run IDs contributing to its (possibly resumed) training
# history. Step numbers do not overlap across runs of the same model, so
# histories are simply merged by step.
MODEL_RUNS = {
    "NoESM_NoUW_baseline": ["r006m9xn"],
    "ESM_NoUW": ["erig627c"],
    "ESM_UW": ["9z78legk", "zt73fiek", "k2ukfze3"],
    "NoESM_UW": ["92b2p3xm", "jmb0ipn8", "hlf2yab8", "0xsxc96i"],
    "ESM_VG": ["mt7codlk", "6rf5hest", "dc4imbr1"],
    "ESM_VG_Contrastive": ["mkn0gjdx", "ikdke5xm", "sfxlen6x", "jh11ik8z"],
    "v4_Projection": ["lvs1v8zp", "trk2biao", "zmalp2a6"],
}

KEYS = [
    "Test/Affinity_Pearson",
    "Test/Affinity_MAE",
    "Test/Contact_AUROC",
    "Test/Contact_AUPRC",
]

OUT_FIELDS = [
    "model",
    "step",
    "Test_Affinity_Pearson",
    "Test_Affinity_MAE",
    "Test_Contact_AUROC",
    "Test_Contact_AUPRC",
]


def main():
    api = wandb.Api()
    all_rows = []

    for model, run_ids in MODEL_RUNS.items():
        combined = {}
        for rid in run_ids:
            try:
                run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
            except Exception as e:
                print(f"  [!] {model} / {rid}: could not open run: {e}")
                continue

            hist = list(run.history(keys=KEYS, samples=500, pandas=False))
            n_added = 0
            for h in hist:
                step = h.get("_step")
                if step is None or h.get("Test/Affinity_Pearson") is None:
                    continue
                combined[step] = h
                n_added += 1
            print(f"  {model} / {rid} ({run.state}): {n_added} points added")

        steps_sorted = sorted(combined.keys())
        print(
            f"{model}: TOTAL {len(steps_sorted)} unique steps, range: "
            f"{min(steps_sorted) if steps_sorted else None} - "
            f"{max(steps_sorted) if steps_sorted else None}"
        )
        for step in steps_sorted:
            h = combined[step]
            all_rows.append(
                {
                    "model": model,
                    "step": step,
                    "Test_Affinity_Pearson": h.get("Test/Affinity_Pearson"),
                    "Test_Affinity_MAE": h.get("Test/Affinity_MAE"),
                    "Test_Contact_AUROC": h.get("Test/Contact_AUROC"),
                    "Test_Contact_AUPRC": h.get("Test/Contact_AUPRC"),
                }
            )

    out_path = "checkpoint_curves_from_wandb.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"WROTE: {out_path}, total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
