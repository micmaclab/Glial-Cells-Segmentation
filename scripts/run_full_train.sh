#!/bin/bash
#SBATCH --partition=gpu4_short     
#SBATCH --gres=gpu:1                  
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                     # Bumped to 48G to comfortably handle 10k x 10k validation images
#SBATCH --time=12:00:00               # 12 hours allows a safe buffer for 250 epochs
#SBATCH --job-name=nnunet_full_5folds
#SBATCH --array=4                   # This automatically creates 5 sub-jobs (Fold 0, 1, 2, 3, 4)
#SBATCH --output=model005/nnunet_fold_%a.log   # Saves separate logs for each fold (e.g., nnunet_fold_0.log)

# 1. Load the pristine GPU Python environment
module load python/gpu/3.10.6-cuda12.9

# 2. Map execution and runtime library paths
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export nnUNet_compile=F

echo "Launching parallel training for 2D config, Fold ${SLURM_ARRAY_TASK_ID} using 250-epoch trainer preset..."

# 3. Run the train command utilizing the built-in 250-epoch trainer class preset
nnUNetv2_train 5 2d ${SLURM_ARRAY_TASK_ID} -tr nnUNetTrainer_250epochs --npz
