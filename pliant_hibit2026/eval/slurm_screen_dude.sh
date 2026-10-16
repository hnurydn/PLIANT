#!/bin/bash
# ==========================================================================
# 13 Eki: dude_brl/litpcba_brl KOPYASI — TEK FARK: code_paths sonuna ablasyon/ eklendi (ABL1/ABL2 modulleri).
# 30 Eyl: BRL (birlesik) kopyasi; MERKEZ=1 -> --center_ligands (ligand merkezi -> cep alfa merkezi, dondurme/cakisma kontrolu yok).
# 28 Eyl kopyasi (vgtop.sh + orjkol/ceptemask code_paths) <- 23 Eyl kopyasi: _esm = lmdb_dude_esm (512) + kapi/ modulleri once aranir
# sumaff_dude_array.sh -- SUMAFF deneyi: ORT vs TOP kolu, DUD-E @ITER (18 Eyl 2026)
# bench24_dude_array.sh'in kopyasi (MERKEZSIZ, 101 hedef, N parca). FARKLAR:
#   - checkpoint: sumaff_dude/ckpt/$ARM/PLANET.iter-$ITER (Anzu'dan aktarildi, md5 dogrulandi)
#   - script: screen_dude.py (ligand basi skor + atom sayisi .npz kaydeder)
#   - cikti: sumaff_dude/out/
#   sbatch --job-name=d24s_TOP --array=0-5 \
#     --export=ALL,ARM=TOP,MODMOD=PLANET_model_gatenorm_sumaff,ITER=100000 sumaff_dude_array.sh
# ==========================================================================
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

SRC=$BASE/datasets/pdbbind2020/dude/${DUDE_LMDB:-lmdb_dude_esm_cafix}   # 23 Eyl: ESM kollari (kapi deneyi)
CKPT=$MY/ckpt/$ARM/PLANET.iter-$ITER
[ -f "$CKPT" ] || { echo "HATA: checkpoint yok: $CKPT"; exit 2; }

SSD=/tmp/hnaydin_d24s_${SLURM_ARRAY_JOB_ID:-$$}_${SHARD}
mkdir -p "$SSD"
cp -r "$SRC/"* "$SSD/"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK  # 22 Eyl: c10 kopyasi
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""   # 22 Eyl: debug GPU dugumlerinde create_var tensoru cuda ya tasiyordu -> sessiz hata

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
