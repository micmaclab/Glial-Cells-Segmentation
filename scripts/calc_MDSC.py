import json
import numpy as np
from pathlib import Path

# Paths to the 5 summary.json files
fold_paths = [
    Path(f"/gpfs/data/ravenlab/micmac/nnUNet_results/Dataset007_BinaryTask/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_{i}/validation/summary.json")
    for i in range(5)
]

all_case_dices = []

for path in fold_paths:
    with open(path, 'r') as f:
        data = json.load(f)
        
    # 'metric_per_case' contains individual evaluation per volume
    for case in data['metric_per_case']:
        # Extract mean Dice across foreground classes for this volume
        metrics = case['metrics']
        # If multi-class, compute average across classes (e.g., class '1', '2', etc.)
        case_dice = np.mean([metrics[cls]['Dice'] for cls in metrics.keys() if cls != '0'])
        all_case_dices.append(case_dice)

# Calculate global mean and standard deviation across all dataset cases
overall_mean_dice = np.mean(all_case_dices)
overall_std_dice = np.std(all_case_dices)

print(f"Overall 5-Fold Mean Dice: {overall_mean_dice:.4f} ± {overall_std_dice:.4f}")