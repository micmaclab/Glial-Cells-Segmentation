import os
import glob
import numpy as np

def get_slice_metrics(npz_path):
    try:
        data = np.load(npz_path)
        probabilities = data['probabilities']
        fg_probs = probabilities[1, 0] # Isolate foreground class 
        
        predicted_fg = fg_probs > 0.5
        uncertain = (fg_probs > 0.2) & (fg_probs < 0.8)
        
        total_pix = int(np.sum(predicted_fg))
        uncertain_pix = int(np.sum(uncertain))
        
        avg_conf = float(np.mean(fg_probs[predicted_fg])) if total_pix > 0 else 0.0
        hesitation = float(uncertain_pix / total_pix) if total_pix > 0 else 0.0
        
        return {
            "pixels": total_pix,
            "avg_conf": avg_conf,
            "hesitation": hesitation
        }
    except Exception as e:
        return None

# Paths setup
dir_d1 = "../nnUNet_results/Dataset001_BinaryTask/prediction_on_test"
dir_d2 = "../nnUNet_results/Dataset002_BinaryTask/prediction_on_test"
dir_d4 = "../nnUNet_results/Dataset004_BinaryTask/prediction_on_test_compareD5"
dir_d5_50 = "../nnUNet_results/Dataset005_BinaryTask/prediction_on_test_50epochs"
dir_d5_250 = "../nnUNet_results/Dataset005_BinaryTask/prediction_on_test_250epochs"


# Gather all unique slice filenames from both directories
files_d1 = {os.path.basename(f) for f in glob.glob(os.path.join(dir_d1, "*.npz"))}
files_d2 = {os.path.basename(f) for f in glob.glob(os.path.join(dir_d2, "*.npz"))}
files_d4 = {os.path.basename(f) for f in glob.glob(os.path.join(dir_d4, "*.npz"))}
files_d5_50 = {os.path.basename(f) for f in glob.glob(os.path.join(dir_d5_50, "*.npz"))}
files_d5_250 = {os.path.basename(f) for f in glob.glob(os.path.join(dir_d5_250, "*.npz"))}

all_unique_slices = sorted(list(files_d4.union(files_d5_50).union(files_d5_250)))

print(f"{'SLICE FILENAME':<35} | {'D1 PIXELS':<10} {'D1 CONF':<8} {'D1 HESIT':<8} | {'D2 PIXELS':<10} {'D2 CONF':<8} {'D2 HESIT':<8} | {'D4 PIXELS':<10} {'D4 CONF':<8} {'D4 HESIT':<8} | {'D5_50 PIXELS':<10} {'D5_50 CONF':<8} {'D5_50 HESIT':<8} | {'D5_250 PIXELS':<10} {'D5_250 CONF':<8} {'D5_250 HESIT':<8}")
print("-" * 130)

for slice_name in all_unique_slices:
    # Process Dataset 1 metrics
    path_d1 = os.path.join(dir_d1, slice_name)
    m_d1 = get_slice_metrics(path_d1) if slice_name in files_d1 else None

    # Process Dataset 2 metrics
    path_d2 = os.path.join(dir_d2, slice_name)
    m_d2 = get_slice_metrics(path_d2) if slice_name in files_d2 else None
    
    # Process Dataset 4 metrics
    path_d4 = os.path.join(dir_d4, slice_name)
    m_d4 = get_slice_metrics(path_d4) if slice_name in files_d4 else None

    # Process Dataset 5 metrics
    path_d5_50 = os.path.join(dir_d5_50, slice_name)
    m_d5_50 = get_slice_metrics(path_d5_50) if slice_name in files_d5_50 else None

    path_d5_250 = os.path.join(dir_d5_250, slice_name)
    m_d5_250 = get_slice_metrics(path_d5_250) if slice_name in files_d5_250 else None

    # Format Dataset 1 strings
    d1_str = f"{m_d1['pixels']:<10} {m_d1['avg_conf']:<8.4f} {m_d1['hesitation']:<8.4f}" if m_d1 else f"{'MISSING':<28}"

    # Format Dataset 2 strings
    d2_str = f"{m_d2['pixels']:<10} {m_d2['avg_conf']:<8.4f} {m_d2['hesitation']:<8.4f}" if m_d2 else f"{'MISSING':<28}"

    # Format Dataset 4 strings
    d4_str = f"{m_d4['pixels']:<10} {m_d4['avg_conf']:<8.4f} {m_d4['hesitation']:<8.4f}" if m_d4 else "MISSING"

    # Format Dataset 5 strings
    d5_50_str = f"{m_d5_50['pixels']:<10} {m_d5_50['avg_conf']:<8.4f} {m_d5_50['hesitation']:<8.4f}" if m_d5_50 else "MISSING"
    d5_250_str = f"{m_d5_250['pixels']:<10} {m_d5_250['avg_conf']:<8.4f} {m_d5_250['hesitation']:<8.4f}" if m_d5_250 else "MISSING"

    # Output aligned row
    print(f"{slice_name:<35} | {d1_str} | {d2_str} | {d4_str} | {d5_50_str} | {d5_250_str}")