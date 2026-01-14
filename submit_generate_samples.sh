#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="gen_samples"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-0
#SBATCH --output=log_generate_samples.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate naip-gen

python sample.py \
    --checkpoint output/checkpoint_530.pt \
    --num_samples 10000 \
    --output_dir generated_samples \
    --batch_size 8