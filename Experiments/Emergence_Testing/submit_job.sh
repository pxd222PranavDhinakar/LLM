#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00

# Verify CUDA setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python test_model.py