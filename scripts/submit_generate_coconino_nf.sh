#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --time=12:00:00
#SBATCH --job-name=coconino_nf_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=log_generate_coconino_nf.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate naip-gen

python generate.py \
    --roi-file polygons/coconino_nf.geojson \
    --gsd 0.3 \
    --train 250000 \
    --test 50000 \
    --val 0 \
    --png \
    coconino_nf