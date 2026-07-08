#!/bin/bash

#SBATCH --partition=gpu4_dev          # Change if needed
#SBATCH --gres=gpu:1                  # Request exactly 1 GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00              # for ensemble
#SBATCH --job-name=nnunet_test_train
#SBATCH --output=model004/postprocessing_determination.log
#SBATCH --nodelist=gn-0002
# 1. Load your Python and Library configuration
module load python/gpu/3.10.6-cuda12.9
#export LD_LIBRARY_PATH=/gpfs/share/apps/python/gpu/3.10.6/lib:$LD_LIBRARY_PATH

# 2. Map your local user bin path so Slurm can find the nnU-Net commands
export PATH=$HOME/.local/bin:$PATH

# 3. Force PyTorch to use its internal bundled cuDNN instead of cluster defaults
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 4. Turn off nnUNet compilation to skip the network plotting issue
export nnUNet_compile=F

# 5. Determine if postprocessing is needed
nnUNetv2_find_best_configuration 4 -c 2d -tr nnUNetTrainer_250epochs
