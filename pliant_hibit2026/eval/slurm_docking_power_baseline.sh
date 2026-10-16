#!/bin/bash
# 15 Oct: docking power + binder-only contact AUROC for the baseline (ABL1), following the
# sb_dp_v2_k700.sh template. Differences: PLANET_BLOSUM=1 (pocket features are converted to
# BLOSUM62 rows), the ablation directory is added to PYTHONPATH, and IS=dp|temas selects the
# job. Smoke test: LIMIT=5.
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 --mem=40G --time=03:50:00
#SBATCH -A hnaydin
#SBATCH --job-name=abl1_dp
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/casf_dock/dp_v2/logs/abl1_dp_%A_%a.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/casf_dock/dp_v2/logs/abl1_dp_%A_%a.err
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL
set -u
ISLER=(dp temas)
IS=${ISLER[$SLURM_ARRAY_TASK_ID]}
KOL=ABL1_P0_700k
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/arf/scratch/hnaydin/PLANET_ORIGINAL/anzu_kod_ikinci/ablasyon:/arf/scratch/hnaydin/PLANET_ORIGINAL/birlesik:/arf/scratch/hnaydin/PLANET_ORIGINAL/vgtop:${PYTHONPATH:-}
export PLANET_BLOSUM=1
echo "=== $KOL IS=$IS BLOSUM=$PLANET_BLOSUM $(hostname) $(date) ==="
L=""; [ -n "${LIMIT:-}" ] && L="--limit $LIMIT"
if [ "$IS" = "dp" ]; then
  O=${LIMIT:+duman_}dp_v2_${KOL}.json
  PLANET_ARCH=$KOL PLANET_OUT=$O python -u casf_dock/dp_v2/eval_docking_power.py $L
else
  PLANET_ARCH=$KOL python -u casf_dock/dp_v2/eval_contact_auroc.py $L
fi
rc=$?
echo "=== Bitti rc=$rc $(date) ==="
exit $rc
