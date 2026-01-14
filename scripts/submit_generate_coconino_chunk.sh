#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --time=12:00:00
#SBATCH --job-name=coconino_chunk_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=log_generate_coconino_chunk.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate naip-gen

python generate.py \
    --roi-file polygons/coconino_chunk.geojson \
    --gsd 0.3 \
    --train 20000 \
    --test 10000 \
    --val 0 \
    --png \
    --num-workers 40 \
    coconino_chunk
