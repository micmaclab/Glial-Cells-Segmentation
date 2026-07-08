#!/bin/bash
  
#SBATCH --partition=gpu4_dev          
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     
#SBATCH --time=02:00:00              
#SBATCH --job-name=d4_conf_level_eval
#SBATCH --output=d4_conf_level.log            

module load python/gpu/3.10.6-cuda12.9

python visualize_confidence.py
