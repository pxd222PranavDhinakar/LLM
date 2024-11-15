#!/bin/bash
#SBATCH --job-name=emergence_analysis    # Job name
#SBATCH --output=emergence_%j.log        # Output file, %j is job ID
#SBATCH --error=emergence_%j.err         # Error file
#SBATCH --time=24:00:00                 # Time limit (24 hours)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=32G                       # Memory per node
#SBATCH --partition=gpu                 # GPU partition

# Load necessary modules
module load GCCcore
module load Python
module load CUDA
module load tqdm
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a

# Print some debugging information
nvidia-smi
which python
python --version

# Run the script
python emergence_analyzer.py