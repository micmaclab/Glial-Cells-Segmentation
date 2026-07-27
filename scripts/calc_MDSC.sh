#!/bin/bash
#SBATCH --partition=radiology
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1    
#SBATCH --time=0-30:00:00
#SBATCH --mem=32G
#SBATCH --output=calc_MDSC_007.out

#python calc_MDSC.py
python calc_MDSC.py
