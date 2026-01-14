#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name="gen_samples"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-0
#SBATCH --output=log_generate_samples.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate naip-gen

python sample.py \
    --checkpoint output/checkpoint_500.pt \
    --num_samples 10000 \
    --output_dir naip_samples \
    --batch_size 8
