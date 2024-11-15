#!/bin/bash
#SBATCH --job-name=emergence_test
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

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python emergence_analyzer.py