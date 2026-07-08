import os
import glob
import numpy as np
import matplotlib.pyplot as plt

dir_d4 = "../nnUNet_results/Dataset004_BinaryTask/prediction_on_test"

def get_slice_metrics(npz_path):
    try:
        data = np.load(npz_path)
        probs = data['probabilities']          # expected [C, D, H, W]
        fg = probs[1, 0]                        # foreground class, singleton depth
        total_px = fg.size

        predicted_fg = fg > 0.5
        uncertain = (fg > 0.2) & (fg < 0.8)

        n_fg = int(predicted_fg.sum())
        return {
            "fg_pixels":      n_fg,
            "conf_committed": float(fg[predicted_fg].mean()) if n_fg else 0.0,  # avg conf where it predicted FG
            "mean_fg_prob":   float(fg.mean()),                                  # overall mean FG probability
            "uncertain_rate": float(uncertain.sum() / total_px),                # true fraction of image in 0.2–0.8
            "fg_probs":       fg[predicted_fg],                                  # for histogram
        }
    except Exception as e:
        print(f"skip {os.path.basename(npz_path)}: {e}")
        return None

files = sorted(glob.glob(os.path.join(dir_d4, "*.npz")))

names, fg_px, conf_c, mean_p, uncert = [], [], [], [], []
all_committed_probs = []

print(f"{'SLICE':<35} | {'FG_PIX':>8} {'CONF_COMMIT':>12} {'MEAN_FG_PROB':>13} {'UNCERT_RATE':>12}")
print("-" * 90)
for f in files:
    m = get_slice_metrics(f)
    if not m:
        continue
    name = os.path.basename(f)
    names.append(name); fg_px.append(m["fg_pixels"])
    conf_c.append(m["conf_committed"]); mean_p.append(m["mean_fg_prob"])
    uncert.append(m["uncertain_rate"])
    all_committed_probs.append(m["fg_probs"])
    print(f"{name:<35} | {m['fg_pixels']:>8} {m['conf_committed']:>12.4f} {m['mean_fg_prob']:>13.4f} {m['uncertain_rate']:>12.4f}")

# Dataset-level summary
if conf_c:
    print("\n=== Dataset004 summary ===")
    print(f"slices                : {len(conf_c)}")
    print(f"avg committed conf    : {np.mean(conf_c):.4f}")
    print(f"avg mean FG prob      : {np.mean(mean_p):.4f}")
    print(f"avg uncertain rate    : {np.mean(uncert):.4f}")
    print(f"total FG pixels       : {sum(fg_px)}")


fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# 1. Per-slice committed confidence, sorted — shows which slices the model is unsure on
order = np.argsort(conf_c)
ax[0,0].bar(range(len(conf_c)), np.array(conf_c)[order], color="#4C72B0")
ax[0,0].axhline(np.mean(conf_c), ls="--", c="k", lw=1, label=f"mean {np.mean(conf_c):.3f}")
ax[0,0].set(title="Committed confidence per slice (sorted)",
            xlabel="slice (sorted)", ylabel="avg conf where FG predicted")
ax[0,0].legend()

# 2. Histogram of all committed foreground probabilities — the confidence distribution
ax[0,1].hist(np.concatenate(all_committed_probs), bins=40, range=(0.5,1.0), color="#55A868")
ax[0,1].set(title="Distribution of foreground probabilities (>0.5)",
            xlabel="probability", ylabel="pixel count")

# 3. Uncertainty rate per slice — where the model "hesitates"
ax[1,0].bar(range(len(uncert)), uncert, color="#C44E52")
ax[1,0].set(title="Uncertain-band rate (0.2–0.8) per slice",
            xlabel="slice", ylabel="fraction of image")

# 4. Does the model get less confident on smaller lesions?
ax[1,1].scatter(fg_px, conf_c, c=uncert, cmap="viridis", s=30)
ax[1,1].set(title="FG size vs confidence (color = uncertainty)",
            xlabel="FG pixel count", ylabel="committed confidence")
cb = fig.colorbar(ax[1,1].collections[0], ax=ax[1,1]); cb.set_label("uncertain rate")

fig.suptitle("Dataset004 — segmentation confidence story", fontsize=14)
fig.tight_layout()
fig.savefig("d4_confidence_story.png", dpi=150)
plt.show()