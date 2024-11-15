#!/bin/bash
#SBATCH --job-name=emergence_analysis
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:RTX2080Ti:1   # Specifically request RTX 2080 Ti
#SBATCH --mem=8gb
#SBATCH --time=24:00:00

# Load necessary modules
module load GCCcore
module load Python
module load CUDA
module load tqdm
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a

# Set CUDA device order
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Print some debugging information
nvidia-smi
which python
python --version

# Run the script
python emergence_analyzer.py