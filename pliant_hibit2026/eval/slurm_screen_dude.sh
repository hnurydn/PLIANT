#!/bin/bash
# DUD-E virtual screening for one PLIANT variant, as a Slurm job array over 101 targets.
#
# MERKEZ=1 selects the CENTRED protocol: --center_ligands moves each ligand's centre onto
# the pocket's alpha-carbon centre, with no rotation and no clash correction. This is the
# protocol reported in the abstract. MERKEZ=0 is the uncentred protocol used elsewhere in
# the project and is NOT what the abstract reports.
#
# The scoring script is screen_dude.py, which also writes per-ligand scores and heavy-atom
# counts to .npz so that size-corrected metrics can be computed afterwards.
#
# Required: ARM (checkpoint folder under sumaff_dude/ckpt/), MODMOD (model module), ITER.
# Example:
#   sbatch --export=ALL,ARM=ABL2_P0_700k,MODMOD=pliant_esm_model,ITER=700000,MERKEZ=1 \
#          slurm_screen_dude.sh
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH -A hnaydin
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix/sumaff_dude/logs/%x_%A_%a.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix/sumaff_dude/logs/%x_%A_%a.err
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix

set -u
NSHARDS=${NSHARDS:-6}
LIMIT=${LIMIT:-0}
ARM=${ARM:?ARM gerekli (ORT|TOP)}
MODMOD=${MODMOD:?MODMOD gerekli}
ITER=${ITER:?ITER gerekli}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
MERKEZ=${MERKEZ:-0}
if [ "$MERKEZ" = "1" ]; then KIP=mrk; CFLAG=--center_ligands; else KIP=msz; CFLAG=""; fi

echo "Basladi: $(date)  Host: $(hostname)  parca=$SHARD/$NSHARDS  ARM=$ARM  ITER=$ITER  MODMOD=$MODMOD  KIP=$KIP"
eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

BASE=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model
EXP=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix
CODE=$EXP/bench19_code
MY=$EXP/sumaff_dude
OUT=$MY/out
mkdir -p "$OUT" "$MY/logs"

SRC=$BASE/datasets/pdbbind2020/dude/${DUDE_LMDB:-lmdb_dude_esm_cafix}   # ESM arms
CKPT=$MY/ckpt/$ARM/PLANET.iter-$ITER
[ -f "$CKPT" ] || { echo "HATA: checkpoint yok: $CKPT"; exit 2; }

SSD=/tmp/hnaydin_d24s_${SLURM_ARRAY_JOB_ID:-$$}_${SHARD}
mkdir -p "$SSD"
cp -r "$SRC/"* "$SSD/"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK  # thread limits, one copy per task
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""   # on debug GPU nodes create_var moved tensors to cuda, which failed silently

python -u "$MY/screen_dude.py" \
    --model_module "$MODMOD" \
    --helper_module DUDE_benchmark_unified \
    --code_paths "/arf/scratch/hnaydin/PLANET_ORIGINAL/kapi,$MY,$CODE,$EXP,$BASE/Scripts,/arf/scratch/hnaydin/PLANET_ORIGINAL/vgtop,/arf/scratch/hnaydin/PLANET_ORIGINAL/orjkol,/arf/scratch/hnaydin/PLANET_ORIGINAL/ceptemask,/arf/scratch/hnaydin/PLANET_ORIGINAL/birlesik,/arf/scratch/hnaydin/PLANET_ORIGINAL/anzu_kod_20261002/ablasyon" \
    --lmdb_dir "$SSD" \
    --model_path "$CKPT" \
    --out_csv "$OUT/dude_${ARM}_${ITER}_${KIP}_shard${SHARD}.csv" \
    --run_name "dude_${ARM}_${ITER}_${KIP}_shard${SHARD}" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --limit_targets "$LIMIT" \
    --device cpu $CFLAG
RC=$?

rm -rf "$SSD"
echo "Bitti: $(date)  rc=$RC"
exit $RC
