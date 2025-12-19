#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="gen_naip"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-0
#SBATCH --output=log_model_train.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate naip-gen

python model.py \
    --dataset_name coconino_nf \
    --batch_size 8 \

