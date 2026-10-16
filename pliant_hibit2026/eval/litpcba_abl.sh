#!/bin/bash
# ==========================================================================
# 13 Eki: dude_brl/litpcba_brl KOPYASI — TEK FARK: code_paths sonuna ablasyon/ eklendi (ABL1/ABL2 modulleri).
# 30 Eyl: BRL (birlesik) kopyasi; MERKEZ=1 -> --center_ligands.
# 28 Eyl: sumaff_dude/litpcba_array_orjcm.sh <- bench24_litpcba_array.sh kopyasi
#   ckpt=sumaff_dude/ckpt/$ARM, cikti sumaff_dude/out/litpcba_$ARM_*, MERKEZSIZ, orjkol/ceptemask/vgtop code_paths
# bench24_litpcba_array.sh -- LIT-PCBA cift kanal @2.400.000, 124 kayit N parcada.
# bench19_litpcba_array.sh'in kopyasi. IKI FARK:
#   1) ITER varsayilani 2400000, cikti dizini bench24/
#   2) --center_ligands YOK  <-- MERKEZSIZ protokol (tezin ana protokolu).
#      bench19 merkezliydi; bu yuzden bench19 <-> bench24 kiyasi
#      protokol-eslesmis DEGILDIR, bench24 kendi icinde kiyaslanir.
#   3) DATA=esm|noesm  -> baseline (PLANET_model_gatenorm) ESM'siz lmdb kullanir.
#
#   sbatch --job-name=l24_esmvggn42 --array=0-5 \
#     --export=ALL,TAG=esmvggn_seed42,CKPTDIR=ESMVGNORM_SUMLN_GN_seed42,\
# MODMOD=PLANET_model_esm2_vector_gating_normalized_sumln_gn bench24_litpcba_array.sh
# ==========================================================================
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
export CUDA_VISIBLE_DEVICES=""   # debug GPU dugumlerinde sessiz cuda hatasi (22 Eyl)
export WANDB_MODE=disabled

python -u "$CODE/LITPCBA_dualchannel.py" \
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
