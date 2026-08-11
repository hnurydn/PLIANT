#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=pose_perturb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/pose_perturb_%j.out
#SBATCH --error=/arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/pose_perturb_%j.err
#SBATCH -A hnaydin
#SBATCH --chdir=/arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model

# Tam üretim koşusu: 271 CASF-2016 core kompleksi, rigid-body mode, RMSD 0/1/2/3/4 Å.
# 5 komplekslik smoke test (2026-08-11) saniyeler sürdü ve kalibrasyon 15/15 tolerans
# içindeydi (bkz. makale/pose_perturbation_teknik_tasarim.md) -> tek çekirdek + debug
# partition (4 saat limit) tam koşu için fazlasıyla yeterli. GPU GEREKMİYOR (RDKit=CPU işi).
#
# Torsiyonel modu da istiyorsanız --mode both yapın (süre yaklaşık 2 katına çıkar,
# yine de debug'ın 4 saatlik limitinin çok altında kalması beklenir).

conda run -p /arf/home/hnaydin/miniconda3/envs/pocketgen python3 \
    /arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_scripts/pose_perturbation.py \
    --core-lmdb /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_planet/core.lmdb \
    --out-dir /arf/scratch/hnaydin/PLANET_ORIGINAL/pose_perturbation_output \
    --chemutils-path /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/Scripts \
    --rmsd-targets 0,1,2,3,4 \
    --mode rigid \
    --n-replicates 1 \
    --seed 42
