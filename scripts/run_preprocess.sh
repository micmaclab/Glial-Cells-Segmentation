#!/bin/bash

#SBATCH --partition=gpu4_short
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=nnunet_preprocess
#SBATCH --output=model011/preprocess_output.log
#SBATCH --exclude=gn-0005,gn-0020,gn-0010,gn-0012,gn-0013,gn-0004


# 1. Load the python environment and patch the libraries
module load python/gpu/3.10.6-cuda12.9
export LD_LIBRARY_PATH=/gpfs/share/apps/python/gpu/3.10.6/lib:$LD_LIBRARY_PATH
# 2. Run the plan and preprocess step

nnUNetv2_plan_and_preprocess -d 11 -c 2d 
