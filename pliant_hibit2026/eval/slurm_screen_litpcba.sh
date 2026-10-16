#!/bin/bash
# LIT-PCBA virtual screening for one PLIANT variant, as a Slurm job array over the 15
# targets (124 target/template pairs).
#
# MERKEZ=1 selects the CENTRED protocol: --center_ligands moves each ligand's centre onto
# the pocket's alpha-carbon centre, with no rotation and no clash correction. This is the
# protocol reported in the abstract. MERKEZ=0 is the uncentred protocol used elsewhere in
# the project and is NOT what the abstract reports; results from the two protocols are not
# comparable with each other.
#
# The scoring script is screen_litpcba.py. It resumes: if the output csv already exists the
# targets it contains are skipped, so an interrupted array task continues where it stopped.
#
# Required: ARM (checkpoint folder under sumaff_dude/ckpt/), MODMOD (model module), ITER.
# Example:
#   sbatch --export=ALL,ARM=BYER0_700k,MODMOD=pliant_esm_pose_model,ITER=700000,MERKEZ=1 \
#          slurm_screen_litpcba.sh
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -A hnaydin
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix/sumaff_dude/logs/%x_%A_%a.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix/sumaff_dude/logs/%x_%A_%a.err
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix

set -u
NSHARDS=${NSHARDS:-6}
LIMIT=${LIMIT:-0}
ARM=${ARM:?ARM gerekli (ckpt/$ARM)}
TAG=$ARM
MODMOD=${MODMOD:?MODMOD gerekli}
DATA=${DATA:-esm}
ITER=${ITER:?ITER gerekli}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
MERKEZ=${MERKEZ:-0}
if [ "$MERKEZ" = "1" ]; then KIP=mrk; CFLAG=--center_ligands; else KIP=msz; CFLAG=""; fi

echo "Basladi: $(date)  Host: $(hostname)  parca=$SHARD/$NSHARDS  TAG=$TAG  ITER=$ITER  DATA=$DATA  LIMIT=$LIMIT  KIP=$KIP"
eval "$(conda shell.bash hook)"
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen

BASE=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model
EXP=/arf/scratch/hnaydin/PLANET_ORIGINAL/sumln_matrix
CODE=$EXP/bench19_code
OUT=$EXP/sumaff_dude/out
mkdir -p "$OUT" "$EXP/sumaff_dude/logs"

if [ "$DATA" = "noesm" ]; then
    SRC=$BASE/datasets/pdbbind2020/LIT-PCBA/lmdb_litpcba_noesm
else
    SRC=$BASE/datasets/pdbbind2020/LIT-PCBA/lmdb_litpcba_esm
fi
[ -d "$SRC" ] || { echo "HATA: veri yok: $SRC"; exit 3; }

CKPT=$EXP/sumaff_dude/ckpt/$ARM/PLANET.iter-$ITER
if [ ! -f "$CKPT" ]; then echo "HATA: checkpoint yok: $CKPT"; exit 2; fi

SSD=/tmp/hnaydin_l28_${SLURM_ARRAY_JOB_ID:-$$}_${SHARD}
mkdir -p "$SSD"
cp -r "$SRC/"* "$SSD/"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""   # avoids the silent cuda failure on debug GPU nodes
export WANDB_MODE=disabled

python -u "$CODE/screen_litpcba.py" \
    --model_module "$MODMOD" \
    --helper_module LITPCBA_benchmark_lmdb \
    --code_paths "/arf/scratch/hnaydin/PLANET_ORIGINAL/kapi,$CODE,$EXP,$BASE/Scripts,/arf/scratch/hnaydin/PLANET_ORIGINAL/vgtop,/arf/scratch/hnaydin/PLANET_ORIGINAL/orjkol,/arf/scratch/hnaydin/PLANET_ORIGINAL/ceptemask,/arf/scratch/hnaydin/PLANET_ORIGINAL/birlesik,/arf/scratch/hnaydin/PLANET_ORIGINAL/anzu_kod_20261002/ablasyon" \
    --lmdb_dir "$SSD" \
    --model_path "$CKPT" \
    --out_csv "$OUT/litpcba_${TAG}_${ITER}_${KIP}_shard${SHARD}.csv" \
    --run_name "litpcba_${TAG}_${ITER}_${KIP}_shard${SHARD}" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --limit_targets "$LIMIT" \
    --device cpu $CFLAG
RC=$?

rm -rf "$SSD"
echo "Bitti: $(date)  rc=$RC"
exit $RC
