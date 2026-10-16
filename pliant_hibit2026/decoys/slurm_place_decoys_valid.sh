#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=120G
#SBATCH --time=3:50:00
#SBATCH --job-name=yer_vt
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/yerlesim/yer_vt_%A_%a.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte/yerlesim/yer_vt_%A_%a.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/cepte

source /arf/home/hnaydin/miniconda3/etc/profile.d/conda.sh
conda activate /arf/home/hnaydin/miniconda3/envs/pocketgen
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=56
export DILIM=150
export NADAY=64
export ZAMANLA=0
export SPLIT=valid_temiz
export MAXDEC=0
SRC=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet_v2
LOC=/tmp/hn_yer4_${SLURM_JOB_ID}
mkdir -p $LOC
t0=$(date +%s)
cp -r $SRC/${SPLIT}.lmdb $LOC/
cp $SRC/${SPLIT}_meta.json $LOC/
echo "LMDB yerel kopya: $(( $(date +%s)-t0 )) sn  boyut: $(du -sh $LOC/${SPLIT}.lmdb | cut -f1)"
export LMDB_DIR=$LOC
echo "SPLIT=$SPLIT  MAXDEC=$MAXDEC  mem=120G  host=$(hostname)  $(date)"
free -g | head -2
python -u place_decoys.py
rc=$?
echo "--- is sonu bellek"
free -g | head -2
rm -rf $LOC
echo "rc=$rc"
exit $rc
