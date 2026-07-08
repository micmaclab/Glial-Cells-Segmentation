#!/bin/bash

#SBATCH --partition=cpu_short
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=nnunet_preprocess
#SBATCH --output=model005/preprocess_output.log


# 1. Load the python environment and patch the libraries
module load python/gpu/3.10.6-cuda12.9
export LD_LIBRARY_PATH=/gpfs/share/apps/python/gpu/3.10.6/lib:$LD_LIBRARY_PATH
# 2. Run the plan and preprocess step

nnUNetv2_plan_and_preprocess -d 5 -c 2d 
