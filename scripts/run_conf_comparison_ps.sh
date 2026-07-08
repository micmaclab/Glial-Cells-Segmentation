#!/bin/bash
  
#SBATCH --partition=gpu4_short         
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     
#SBATCH --time=01:00:00              
#SBATCH --job-name=single_slice_eval
#SBATCH --output=single_slice_results.log
#SBATCH --nodelist=gn-0005            

module load python/gpu/3.10.6-cuda12.9

python compute_single_slice_confidence.py
