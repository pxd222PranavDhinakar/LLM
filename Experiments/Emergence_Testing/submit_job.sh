#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00

module purge
module load CUDA/11.7.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load tqdm

python test_model.py