#!/bin/bash
#SBATCH --partition=barbun-cuda
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/kont_truba/logs/%x_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/kont_truba/logs/%x_%j.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/kont_truba
# ═══ 2 Eki 2026: Anzu kollarinin TRUBA'da DEVAMI (Anzu run_byer.sh / run_kont.sh ile BIREBIR) ═══
#   TIP=byer -> train_pliant_esm_pose.py  + pliant_esm_pose_train_model  (PYTHONPATH Y:G:BR:B:V:K:fixed_run)
#   TIP=esmuw -> [ABLASYON KOL 2, 2 Eki] ablasyon/train_pliant.py + pliant_esm_model
#               (ESM+UW; vector attention + gating YOK; PYTHONPATH ABL:Y:G:BR:B:V:K:fixed_run)
#   TIP=blosum -> [ABLASYON KOL 1] Baseline+UW: kol 2 + rezidu girisi BLOSUM 20 (veri ..._byer_blosum20)
#   Kod: $KOD = Anzu'dan md5 dogrulamali kopya (anzu_kod_20261002). Veri: byer LMDB'nin Anzu md5'li kopyasi.
#   ⛔ Egitim betiklerindeki SABIT TRUBA sys.path eklemeleri (Planet_model/Scripts, PLANET_FIXED_20260917)
#      Anzu'da HIC calismiyordu (klasorler yok) -> TRUBA kopyasinda DEVRE DISI; kapi her modulu md5'le sinar.
#   DEVAM: D icindeki en son TAM ckpt (N % 20000 == 0, optimizer dahil) bulunur; yoksa BAS_CKPT/BAS_ADIM.
#   Ardisik is (afterany) ayni betikle kaldigi yerden surer; N >= MAXSTEP ise hemen cikar.
#   DUMAN=1 -> wandb kapali, ayri klasor, 15 dk sonra durur (wandb run'i kirletmeden dogrulama).
# Env: TIP AD POZ BAS_CKPT BAS_ADIM WANDB_ID [MAXSTEP=500000] [DUMAN=0] [SEED=42] [MIN_DECOY=11]  (5 Eki: seed parametresi, varsayilan 42 = eski davranis)
#   BAS_CKPT=YOK + BAS_ADIM=0 -> SIFIRDAN egitim (2 Eki karari: kontrastif kollar TRUBA'da bastan).
#   WANDB_ID=YENI -> yeni wandb run (id egitim betigince olusturulur, D/wandb_run_id.txt'ye yazilir).
set -u
: "${TIP:?}"; : "${AD:?}"; : "${POZ:?}"; : "${BAS_CKPT:?}"; : "${BAS_ADIM:?}"; : "${WANDB_ID:?}"
MAXSTEP=${MAXSTEP:-500000}; DUMAN=${DUMAN:-0}
R=/arf/scratch/hnaydin/PLANET_ORIGINAL
KOD=$R/anzu_kod_20261002
P=$R/Planet_model/datasets/pdbbind2020/lmdb_planet_v2_cepte_prot_byer_anzu2
OZN=512   # beklenen rezidu oznitelik boyutu (veri kapisi)
MD5=$R/kont_truba/aktarim/anzu_kod_md5.txt:$R/kont_truba/aktarim/ablasyon_md5.txt   # ':' ile birden cok liste
ABL=$KOD/ablasyon; KT=$KOD/kontrastif; Y=$KOD/binderyer; G=$KOD/gerceketiket; BR=$KOD/birlesik; B=$KOD/bxc; V=$KOD/vgtop; K=$KOD/kapi; F=$KOD/fixed_run
case $TIP in
  byer) MOD=pliant_esm_pose_train_model; BETIK=$B/train_pliant_esm_pose.py; PP=$Y:$G:$BR:$B:$V:$K:$F; KONT=0; KARG="" ;;
        KARG="--kont_lambda 1.0 --kont_tau 0.1" ;;
  esmuw) MOD=pliant_esm_model; BETIK=$ABL/train_pliant.py; PP=$ABL:$Y:$G:$BR:$B:$V:$K:$F; KONT=0; KARG="" ;;
  blosum) MOD=pliant_model; BETIK=$ABL/train_pliant.py; PP=$ABL:$Y:$G:$BR:$B:$V:$K:$F; KONT=0; KARG=""
          P=$R/Planet_model/datasets/pdbbind2020/lmdb_planet_v2_cepte_prot_byer_blosum20; OZN=20
          [ -f $P/DONUSUM.json ] || { echo "HATA: $P/DONUSUM.json yok"; exit 3; } ;;
  *) echo "HATA TIP"; exit 2 ;;
esac
if [ "$DUMAN" = "1" ]; then D=$R/kont_truba/kosu/${AD}_duman; else D=$R/kont_truba/kosu/$AD; fi
echo "=== TRUBA DEVAM $AD TIP=$TIP POZ=$POZ SEED=${SEED:-42} MIN_DECOY=${MIN_DECOY:-11} DUMAN=$DUMAN MAXSTEP=$MAXSTEP : $(date) host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
[ -f $P/train.lmdb/data.mdb ] || { echo "HATA: veri yok $P"; exit 3; }
for m in ${MD5//:/ }; do [ -f $m ] || { echo "HATA: md5 listesi yok $m"; exit 3; }; done
mkdir -p $D
if [ "$WANDB_ID" != "YENI" ]; then
  [ -f $D/wandb_run_id.txt ] || echo -n "$WANDB_ID" > $D/wandb_run_id.txt
  [ "$(cat $D/wandb_run_id.txt)" = "$WANDB_ID" ] || { echo "HATA: wandb id uyusmuyor"; exit 3; }
fi
[ -f $D/wandb_run_id.txt ] && echo "wandb id: $(cat $D/wandb_run_id.txt)"
# en son TAM ckpt
SON=$(ls $D 2>/dev/null | sed -n 's/^PLANET\.iter-\([0-9]*\)$/\1/p' | awk '$1 % 20000 == 0' | sort -n | tail -1)
if [ -n "$SON" ] && [ "$SON" -gt "$BAS_ADIM" ]; then CK=$D/PLANET.iter-$SON; ADIM=$SON; else CK=$BAS_CKPT; ADIM=$BAS_ADIM; fi
[ "$ADIM" -ge "$MAXSTEP" ] && { echo "=== zaten bitti ($ADIM >= $MAXSTEP) ==="; exit 0; }
if [ "$CK" = "YOK" ]; then
  [ "$ADIM" = "0" ] || { echo "HATA: ckpt YOK ama adim $ADIM"; exit 3; }
  DEVAM_ARG=""; echo "SIFIRDAN egitim"
else
  [ -f "$CK" ] || { echo "HATA: ckpt yok $CK"; exit 3; }
  DEVAM_ARG="--load_checkpoint $CK --initial_step $ADIM"
  echo "devam: $CK  adim=$ADIM  md5=$(md5sum $CK | cut -c1-8)"
fi
source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
export PYTHONPATH=$PP
export DECOY_ETIKET=gercek BINDER_POZ_ORAN=$POZ KONT_GORUNUM=$KONT
export KOL_ADI=${AD}_${ADEK:-TRUBA}   # devam eden Anzu run'lari: ADEK=ANZU (wandb adi degismesin)
if [ "$DUMAN" = "1" ]; then export WANDB_MODE=disabled; else export WANDB_MODE=online; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# temiz calistirma klasoru (sys.path[0] golgelemesine karsi) + SABIT TRUBA yollari devre disi
KOSU=$D/_kosu_$SLURM_JOB_ID; mkdir -p $KOSU
sed -e 's|^    sys.path.insert(0, scripts_path)|    pass  # [TRUBA=ANZU] devre disi|' \
    -e 's|^    sys.path.insert(0, normalized_path)|    pass  # [TRUBA=ANZU] devre disi|' \
    -e 's|^    if os.path.isdir(_p):|    if False and os.path.isdir(_p):  # [TRUBA=ANZU] devre disi|' \
    $BETIK > $KOSU/$(basename $BETIK)
[ $(grep -c "TRUBA=ANZU" $KOSU/$(basename $BETIK)) -eq 3 ] || { echo "HATA: yol yamasi tutmadi"; exit 8; }
diff $BETIK $KOSU/$(basename $BETIK)
# SERT KAPI: yuklenen HER proje modulu KOD altindan ve Anzu md5'iyle ayni mi?
cat > $KOSU/_dogrula.py <<'PYX'
import os, sys, hashlib, importlib
betik = sys.argv[1]
src = open(betik, encoding='utf-8').read().split('def move_to_device')[0]
exec(compile(src, betik, 'exec'))          # egitim betiginin import blogu (yol eklemeleri dahil)
importlib.import_module(os.environ['MOD']); import margin_losses
if os.environ['TIP'] == 'kont': import kont_kayip
KOD = os.environ['KOD']
md5 = {}
for liste in os.environ['MD5L'].split(':'):
    for l in open(liste):
        h, f = l.split(None, 1); md5[os.path.join(KOD, f.strip())] = h
hata = 0; n = 0
# 3 Eki: TRUBA'da conda ortami da /arf altinda (/arf/home/.../envs/pocketgen) -> stdlib'i atla;
# __main__/__mp_main__ = bu kapinin kendisi. (Anzu'da ortam /media disinda oldugu icin gorulmemisti.)
_std = tuple(sorted({os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}))
for ad, m in sorted(sys.modules.items()):
    f = getattr(m, '__file__', None)
    if ad in ('__main__', '__mp_main__'): continue
    if not f or 'site-packages' in f or not f.startswith('/arf'): continue
    if os.path.realpath(f).startswith(_std): continue
    f = os.path.abspath(f); n += 1
    if f not in md5:
        print('⛔ KOD DISI modul:', ad, f); hata += 1; continue
    if hashlib.md5(open(f, 'rb').read()).hexdigest() != md5[f]:
        print('⛔ md5 FARKLI:', ad, f); hata += 1
    else:
        print('  ok', ad, f.replace(KOD + '/', ''))
d = sys.modules['PLANET_datautils_esm2_vector_cokcep']
print('BINDER_POZ_ORAN', d.BINDER_POZ_ORAN, '| DECOY_ETIKET', d.DECOY_ETIKET, '| KONT_GORUNUM', getattr(d, 'KONT_GORUNUM', 'YOK'))
if abs(d.BINDER_POZ_ORAN - float(os.environ['BINDER_POZ_ORAN'])) > 1e-9 or d.DECOY_ETIKET != 'gercek': hata += 1
if os.environ['TIP'] == 'kont' and d.KONT_GORUNUM is not True: hata += 1
if os.environ['TIP'] in ('byer', 'esmuw', 'blosum') and getattr(d, 'KONT_GORUNUM', False): hata += 1
if os.environ['TIP'] in ('esmuw', 'blosum'):
    L = sys.modules.get('layers_pliant_esm')
    if L is None or os.path.dirname(L.__file__) != os.path.join(KOD, 'ablasyon'): print('⛔ layers_pliant_esm yuklenmedi/yanlis'); hata += 1
# VERI KAPISI: ilk 5 cebin rezidu oznitelik boyutu beklenenle ayni mi? (chemutils sessizce kirpar/tamamlar!)
import lmdb, pickle, json as _j
import numpy as _np
_k = [r['key'] for r in _j.load(open(os.environ['VERI'] + '/train_meta.json'))][:5]
with lmdb.open(os.environ['VERI'] + '/train.lmdb', readonly=True, lock=False).begin() as _t:
    _b = sorted({_np.asarray(_r.get_feature()).size for k in _k for _r in pickle.loads(bytes(_t.get(k.encode()))).pocket_residues})
print('veri rezidu oznitelik boyutu:', _b, '| beklenen', os.environ['OZN'])
if _b != [int(os.environ['OZN'])]: print('⛔ VERI-MODEL BOYUT UYUSMAZ'); hata += 1
if os.environ['TIP'] == 'blosum':
    import chemutils_esm2_vector as _cu
    if _cu.FEATURE_DIM != 20: print('⛔ FEATURE_DIM', _cu.FEATURE_DIM, '!= 20'); hata += 1
if n < 6: print(f'⛔ yalniz {n} proje modulu sinandi (beklenen >= 6) -- filtre fazla genis'); hata += 1
print(f'{n} proje modulu sinandi, hata={hata}')
sys.exit(9 if hata else 0)
PYX
cd $KOSU
TIP=$TIP MOD=$MOD KOD=$KOD MD5L=$MD5 VERI=$P OZN=$OZN python -u $KOSU/_dogrula.py $KOSU/$(basename $BETIK) || { echo "=== KAPI KALDI ==="; exit 9; }
echo "✅ kapi gecti"
# veriyi dugum diskine kopyala (meta JSON dahil; yol adinda 'esm2' OLMAMALI -> ckpt adi PLANET.iter-*)
# 2 Eki: bellek olcumu (Anzu, ayni tarif): PSS toplam 9 GB, ana surec tepe 5 GB -> 48G = 9 + LMDB sayfa onbellegi ~13 + pay
LOC=/tmp/hnaydin_kont_${SLURM_JOB_ID}
trap 'rm -rf $LOC' EXIT          # walltime / scancel / hata: dugumde 13 GB cop birakma
echo "/tmp: $(df -hT /tmp | tail -1)"
[ "$(df -T /tmp | tail -1 | awk '{print $2}')" = "tmpfs" ] && echo "⚠️ /tmp tmpfs (RAM) -> 13 GB bellekten sayilir"
bos=$(df -Pk /tmp | tail -1 | awk '{print $4}')
[ "$bos" -gt 20000000 ] || { echo "HATA: /tmp'de yer yok ($((bos/1024/1024)) GB < 20 GB)"; exit 4; }
mkdir -p $LOC; t0=$(date +%s)
cp -r $P/train.lmdb $P/valid_temiz.lmdb $P/core.lmdb $LOC/ && cp $P/train_meta.json $P/valid_temiz_meta.json $P/core_meta.json $LOC/ || { echo "HATA: kopyalama basarisiz"; exit 4; }
for f in train.lmdb/data.mdb valid_temiz.lmdb/data.mdb core.lmdb/data.mdb train_meta.json valid_temiz_meta.json core_meta.json; do
  [ "$(stat -c %s $P/$f)" = "$(stat -c %s $LOC/$f)" ] || { echo "HATA: boyut uyusmuyor $f"; exit 4; }
done
echo "kopyalama $(( $(date +%s)-t0 )) sn, boyutlar kaynakla ayni"; du -sh $LOC
if [ "$DUMAN" = "1" ]; then   # tam is wandb ONLINE acacak -> erisim simdiden sinanir
  python -c "import urllib.request as u, urllib.error as e, sys
try: u.urlopen('https://api.wandb.ai', timeout=15)
except e.HTTPError: pass
except Exception: sys.exit(1)" && echo "wandb erisimi: OK" || { echo "⛔ wandb sunucusuna erisim YOK (tam is online acamaz)"; exit 6; }
fi
ZAMAN=""; [ "$DUMAN" = "1" ] && ZAMAN="timeout 900"
$ZAMAN python -u $KOSU/$(basename $BETIK) -t $LOC/train.lmdb -v $LOC/valid_temiz.lmdb -te $LOC/core.lmdb -d $D \
  --seed ${SEED:-42} --model_module $MOD --n_pocket 4 --n_decoy 2 --min_decoy ${MIN_DECOY:-11} --olcek 1 --clip_norm 200 \
  $DEVAM_ARG \
  --anneal_iter 20000 --save_iter 20000 --print_iter 200 --max_step $MAXSTEP --sik_adim 5000 --sik_ckpt 1000 \
  --mol_marj_lambda 1.0 --mol_marj 1.0 \
  --marj_lambda 1.0 --marj 1.0 --marj_oteleme_min 1.0 --marj_oteleme_max 3.0 --marj_her_n 1 \
  $KARG \
  --zor_dosya $P/train_zorluk.pkl --zor_ust 60
rc=$?
if [ "$DUMAN" = "1" ]; then
  OUT=/arf/scratch/hnaydin/PLANET_ORIGINAL/kont_truba/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
  nh=$(grep -c "Adımı Hatası" $OUT); np=$(grep -c "^\[[0-9]" $OUT)
  echo "DUMAN: rc=$rc adim_hatasi=$nh print_satiri=$np"
  if [ "$rc" = "124" ] && [ "$nh" = "0" ] && [ "$np" -ge 1 ]; then echo "✅ DUMAN GECTI"; rc=0; else echo "⛔ DUMAN KALDI"; [ "$rc" = "0" ] && rc=1; fi
fi
echo "=== Bitti (rc=$rc): $(date) ==="
exit $rc
