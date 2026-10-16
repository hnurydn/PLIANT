#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=dp_v2_uw700
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/casf_dock/dp_v2/logs/dp_v2_%A_%a.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/casf_dock/dp_v2/logs/dp_v2_%A_%a.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL
# Dizi indeksi -> model etiketi (eval_docking_power.py MODELS sirasi). Duman testi: LIMIT=3 ve --array ile birkac indeks.
KOLLAR=(ABL2_P0_700k)
KOL=${KOLLAR[$SLURM_ARRAY_TASK_ID]}
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/arf/scratch/hnaydin/PLANET_ORIGINAL/anzu_kod_ikinci/ablasyon:/arf/scratch/hnaydin/PLANET_ORIGINAL/birlesik:/arf/scratch/hnaydin/PLANET_ORIGINAL/vgtop:$PYTHONPATH   # birlesik modulu (layers_pliant_esm_pose ayni dizinde)
echo "=== $KOL  $(hostname)  $(date) ==="
if [ -n "$LIMIT" ]; then
  PLANET_ARCH=$KOL PLANET_OUT=duman_${KOL}.json python -u casf_dock/dp_v2/eval_docking_power.py --limit $LIMIT
else
  PLANET_ARCH=$KOL python -u casf_dock/dp_v2/eval_docking_power.py
fi
rc=$?
echo "=== Bitti rc=$rc $(date) ==="
exit $rc
