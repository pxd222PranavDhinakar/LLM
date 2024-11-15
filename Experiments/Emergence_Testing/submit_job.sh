#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00

module purge
module load Python/3.10.4
module load CUDA/11.7.0 
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0  
export PYTHONPATH="/home/pxd222/LLM/Experiments:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

cd /home/pxd222/LLM/Experiments/Emergence_Testing

# Run script
srun python test_model.py