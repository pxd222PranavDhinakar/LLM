#!/bin/bash
#SBATCH --job-name=emergence_analysis
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00

# Load necessary modules
module purge  # Clear any existing modules
module load GCC/11.3.0
module load CUDA/11.7.0
module load PyTorch/2.0.1-foss-2022b-CUDA-11.7.0
module load matplotlib/3.7.1
module load Seaborn/0.12.2

export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=1

# Run the script
python emergence_analyzer.py