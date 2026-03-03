#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name="eval_naip"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-4
#SBATCH --output=log_evaluate.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate naip-gen

python evaluate.py \
    --real_dir data/coconino_chunk/test/ \
    --generated_dir naip_samples/ \
    --output_dir eval_results \
    --batch_size 64
