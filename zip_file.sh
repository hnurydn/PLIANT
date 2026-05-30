#!/bin/bash
#SBATCH --partition=barbun
#SBATCH --cpus-per-task=40
#SBATCH --job-name=zip_ssd_optimized
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/arf/scratch/hnaydin/zip_%j.out
#SBATCH --error=/arf/scratch/hnaydin/zip_%j.err
#SBATCH -A hnaydin

cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/lmdb_tensor/

tar --sparse -czf train_tensor.tar.gz train_tensor.lmdb train_tensor_meta.json &
tar --sparse -czf valid_tensor.tar.gz valid_tensor.lmdb valid_tensor_meta.json &
tar --sparse -czf core_tensor.tar.gz  core_tensor.lmdb  core_tensor_meta.json &
wait
echo "Tamamlandi"
# cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/

# tar -I pigz -cf all_pickles.tar.gz all_pickles

# cd /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/

# tar -cf /arf/scratch/hnaydin/PLANET_ORIGINAL/Planet_model/datasets/pdbbind2020/planet_esm2_pkl.tar planet_esm2_pkl
# zip -r pkls.zip pkls
# zip -r 2001-2010.zip 2001-2010
# zip -r 2011-2019.zip 2011-2019
## unzip 2011-2019.zip
