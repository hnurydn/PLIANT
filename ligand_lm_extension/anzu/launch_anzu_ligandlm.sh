#!/bin/bash
# --- launch_anzu_ligandlm.sh ---
# Anzu'da (Slurm YOK, dogrudan calistiriliyor) 2 LigandLM modelini AYNI TEK
# GPU'da baslatir (kullanici karari: birden fazla GPU almak baskalarinin
# ise erisimini kisitlayabilir, o yuzden sadece TEK bos GPU kullaniliyor).
# GPU 2 secildi (RTX 5000 Ada, 32GB VRAM, 2026-08-12 itibariyla tamamen bos)
# -- GPU 0/1/6 baska kullanicilarin isleriyle dolu, ONLARA DOKUNULMUYOR.
# 3. model (v4_Projection) TRUBA'ya gonderiliyor (bkz. launch_v4proj_ligandlm.sh).
#
# Kullanim: bash launch_anzu_ligandlm.sh
set -e

GPU=2
SCRIPTS=/home/havvanur/Scripts
DATA=/media/ubuntu/8TB/havvanur/lmdb_planet_v3   # <-- transfer sonrasi burada olmali
LOGDIR=/home/havvanur/ligandlm_logs
mkdir -p "$LOGDIR"

source /home/havvanur/miniconda3/etc/profile.d/conda.sh
conda activate havvanur_thesis

if [ ! -d "$DATA" ]; then
    echo "HATA: $DATA bulunamadi. lmdb_planet_v3 once TRUBA'dan transfer edilmeli."
    exit 1
fi

run_model () {
    local script=$1
    local save_dir=$2
    local log_name=$3

    echo "Baslatiliyor: $script (GPU $GPU) -> $log_name"
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 "$SCRIPTS/$script" \
        -t "$DATA/train.lmdb" -v "$DATA/valid.lmdb" -te "$DATA/core.lmdb" \
        -d "$save_dir" --seed 42 --decoy_count 10 \
        > "$LOGDIR/${log_name}.out" 2> "$LOGDIR/${log_name}.err" &
    disown
    echo "  PID: $!"
}

# GPU 2 (RTX 5000 Ada, 32GB) -- 2 model AYNI GPU'da birlikte
run_model PLANET_train_esm2_vector_gating_ligandlm.py \
    /home/havvanur/ESM2_UW_VECTOR_GATING_LigandLM esmvg_ligandlm

run_model PLANET_train_baseline_ligandlm.py \
    /home/havvanur/noESM_noUW_LigandLM baseline_ligandlm

echo ""
echo "2 model GPU $GPU'da baslatildi. Takip icin:"
echo "  tail -f $LOGDIR/*.out"
echo "  nvidia-smi  (GPU $GPU'da 2x python process gormelisin, 0/1/6'ya dokunulmadi)"
echo "  ps aux | grep PLANET_train"
