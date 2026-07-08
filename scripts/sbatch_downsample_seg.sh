#!/bin/bash
#SBATCH --partition=radiology
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1    
#SBATCH --time=0-14:00:00
#SBATCH --mem=128G
#SBATCH --output=downsample_seg.out

python downsample_segmentations.py
