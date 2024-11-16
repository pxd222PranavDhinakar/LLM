#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00


module purge
module load tqdm
module load Python/3.10.4
module load CUDA/11.7.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

# Run script
srun python emergence_analyzer.py