"""
Stage and upload the 7 trained-model checkpoints (step 1,900,000, the "best
checkpoint" selected in the thesis / used throughout the JCIM manuscript) to
the existing Hugging Face dataset repo, alongside the LMDB training data.

Run on the TRUBA server (conda env `pocketgen`, which already has
huggingface_hub + hf_transfer installed). Requires HF_TOKEN env var
(write-access token) to be set for this run only.

Usage:
    HF_TOKEN=hf_xxx python hf_upload_checkpoints.py
"""

import os
import shutil
import time

try:
    import hf_transfer  # noqa: F401

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    fast = True
except Exception:
    fast = False

from huggingface_hub import HfApi
PLANET_ROOT = os.environ.get("PLANET_ROOT", ".")  # root of the cloned repo / data checkout



TOKEN = os.environ.get("HF_TOKEN")
REPO = "hnurydn/PLANET-datasets"
PATH_IN_REPO = "checkpoints_article_pose_dependence"

BASE = PLANET_ROOT
STAGE = "/arf/scratch/hnaydin/PLANET_ORIGINAL/hf_checkpoint_staging"

# source dir (relative to BASE) -> canonical model name used throughout the
# manuscript / checkpoint_curves_from_wandb.csv. The two ESM_VG_Contrastive
# entries are two independently trained runs of the same architecture
# (naming on disk: 1b11d / 1b11d_2); kept distinct rather than merged since
# it is not confirmed which one (if either) is the exact checkpoint behind
# each derived DUD-E/LIT-PCBA "v2_*" ensemble-variant CSV in
# statistical_analysis/ — see README note.
CHECKPOINTS = {
    "noESM_noUW_1:11_512": "NoESM_NoUW_baseline",
    "esm_nouw": "ESM_NoUW",
    "ESM2_UW_1:11": "ESM_UW",
    "noESM_UW_1:11_512": "NoESM_UW",
    "ESM2_UW_VECTOR_GATING_1:11": "ESM_VG",
    "ESM2_VG_CONTRASTIVE_1b11d": "ESM_VG_Contrastive_run1",
    "ESM2_VG_CONTRASTIVE_1b11d_2": "ESM_VG_Contrastive_run2",
    "ESM2_VG_CONTRASTIVE_v4_projection": "v4_Projection",
}

STEP = "1900000"


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def main():
    if not TOKEN:
        raise SystemExit("HF_TOKEN not set")

    os.makedirs(STAGE, exist_ok=True)
    for src_dir, model_name in CHECKPOINTS.items():
        src = os.path.join(BASE, src_dir, f"PLANET.iter-{STEP}")
        if not os.path.isfile(src):
            log(f"  [!] MISSING: {src}")
            continue
        dst = os.path.join(STAGE, f"{model_name}.iter-{STEP}.pt")
        shutil.copyfile(src, dst)
        log(f"  staged {model_name}: {os.path.getsize(dst) / 1e6:.1f} MB")

    log(f"START upload  hf_transfer={fast}")
    api = HfApi(token=TOKEN)
    api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=STAGE,
        path_in_repo=PATH_IN_REPO,
        commit_message="Add trained checkpoints (step 1.9M) for pose-dependence article",
    )
    log("UPLOAD DONE")


if __name__ == "__main__":
    main()
