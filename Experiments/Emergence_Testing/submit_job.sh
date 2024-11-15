#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00

module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.7.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load tqdm

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Verify CUDA setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python test_model.py