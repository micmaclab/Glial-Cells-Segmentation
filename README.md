# Microglia Segmentation Pipeline

This is a step-by-step tutorial for recreating our Microglia Mapping model - an nnU-Net 2D binary segmentation model for microglia. It walks through the
full pipeline.

 [Project presentation](https://canva.link/ahanvcmftd3r573)

---

## Background

Microglia are the only immune cells residing in the brain parenchyma - the main
working tissue of the brain. Once dismissed as mere "support cells," they're now
recognized as active modulators of neural circuits: their morphology encodes
functional state, meaning they're dynamic, not static. They survey the tissue,
prune synapses, respond to injury, and shape brain dynamics in real time, and
there are countless of them, distributed across space like a matrix. They're
effectively the cleaning cells of the central nervous system.

That density and complexitys makes them hard to study
computationally. They're small and densely packed, so distinguishing individual
cells without computational extraction is difficult. Manual tracing is
infeasible at whole-brain volume. Classical (global) thresholding fails because
intensity varies substantially across a slide, such as gray matter vs. white matter can shift the baseline, and quantifying glial organization more broadly is
difficult given how complex and varied their spatial distribution and morphology
are.

Published ML segmentation models exist for mouse brains (e.g. Stain AI, which
classifies cell shapes/features in mouse tissue), but not for primate brains, and
none of the mouse models are publicly available. This project closes that gap:
we want to provide a reusable, open-source deep learning pipeline that segments microglia across a
**whole Macaque brain**, which is the first to do so at this scale.

---

## Data

**The images and labels this pipeline was built on are not included here and are
not public.** You'll need your own histology images before any step below will
run (this repo only has the scripts, not the data).

For reference, here's what the original data looked like, so you have a gauge of
what "similar enough to use this pipeline as-is" means. It's **two distinct
sets**, not one:
- **Set 1 (whole-brain coverage):** 170 IBA1-stained slices (40 µm section
  thickness) spanning a whole healthy Macaque brain, scanned at 2 µm/pixel.
  Whole-slide, single-channel (grayscale) histology sections, roughly
  28,000 × 34,000 px per slide. We call this "low-res" only relative to Set 2
  below - 2 µm/pixel is itself a high-resolution scan.
- **Set 2 (fine morphology):** 25 "high-res" 0.25 µm/pixel slices of particular
  glia clusters/morphologies. Used specifically to teach the model finer
  microglial process/shape detail during training (see step 6).
- Stored in TIFF (`.tiff`) format, one file per section.
- Target structure: microglia, segmented as a binary mask (background vs.
  microglia).

If you'd like access to the original images/labels, or want to check whether your
own data is a good match for this pipeline, contact:
- **Bradley Karat** — Bradley.Karat@nyulangone.org
- **Erika Raven** — Erika.Raven@nyulangone.org

---

## 0. Environment setup

**A. General image processing packages** (numpy, cv2,
nibabel, tifffile, scipy, scikit-image). Create your own conda env once:
```bash
conda create -n microglia_seg python=3.10 -y #Python version 3.10+ is preferred but 3.4+ should work as well
conda activate microglia_seg
pip install numpy opencv-python nibabel h5py tifffile scipy scikit-image
```

Then for every job/session that needs this env:
```bash
conda activate microglia_seg
```

**B. nnU-Net training/inference/preprocessing** (GPU, torch, nnunetv2). This
project was built on a cluster with a shared GPU Python module (`module load
python/gpu/3.10.6-cuda12.9`) as the base - **that module name is specific to this
cluster.** If you're on a different machine:
- Check what your own cluster/system offers first: `module avail python` /
  `module spider cuda` (naming varies a lot between institutions), or ask your
  cluster's docs/admins what the equivalent GPU-enabled Python module is called.
- If there's no such module (or you're not on a shared cluster at all), skip
  `module load` entirely and just build your own GPU-capable environment instead -
  a conda env or venv with a CUDA build of torch matching your GPU driver (see
  [pytorch.org's install matrix](https://pytorch.org/get-started/locally/) to pick
  the right command for your CUDA version), then install nnU-Net into that same
  env. Everything below (`nnUNet_compile`, the console scripts, `LD_LIBRARY_PATH`)
  works the same either way - it's only the "how do I get a working GPU Python"
  step that's cluster-specific.

Whichever base you end up with:
```bash
module load <yourchosenmodule>
export PATH=$HOME/.local/bin:$PATH
export nnUNet_compile=F   # skips a network-plotting bug
```
If nnU-Net itself isn't installed yet — follow the official install instructions at
**https://github.com/MIC-DKFZ/nnUNet** (README's "Installation" section) to install
nnU-Net v2 and a matching GPU build of torch into whichever base you set up above.
Once installed, the `nnUNetv2_*` console commands used throughout
this pipeline (`nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, `nnUNetv2_predict`,
etc.) should be on your `PATH`. You'll also need:
```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```
(adjust the python version in that path to whatever nnU-Net's install actually used).

**Required for every nnU-Net command**: three path variables nnU-Net needs to know
where to read/write data. None of the sbatch scripts export these explicitly, which
only works if they're already set in your `~/.bashrc` (Slurm's `sbatch` inherits the
submitting shell's environment by default). Set them once, e.g. by adding to
`~/.bashrc`:
```bash
export nnUNet_raw=<PathToYourWorkingFolder>/nnUNet_raw
export nnUNet_preprocessed=<PathToYourWorkingFolder>/nnUNet_preprocessed
export nnUNet_results=<PathToYourWorkingFolder>/nnUNet_results
```

> **Note:** the sbatch scripts hardcode a dataset ID, an output log directory
> (e.g. `model001/...`), and sometimes a `--array` range or `--nodelist`. These are
> reused/edited by hand between runs — before launching, check that they point at
> the dataset/model number you actually intend to run, not whatever the last person
> left in the file.

---

## 1. Generate groundtruth labels via intensity thresholding

**Script:** `microglia_thresholding.py` (submit via `sbatch_microglia_thresholding.sh`)

**Noted**: Remember to check if the directory in all of these scripts needs to be changed to your directory (e.g. `in_dir = /gpfs/data/ravenlab/micmac`) as in the scripts themselves.


For each of the 170 full-resolution 2 µm slides (listed in `microglia_filenames.txt`):
- CLAHE-enhances contrast (`cv2.createCLAHE`) for a cleaner visual reference.
- Finds a representative "white matter" reference patch (500×500 px grid, picks the
  tile with highest masked mean intensity) and computes an Otsu threshold on it.
- Locally re-thresholds the rest of the slide in 50×50 px blocks, histogram-matching
  each block to the reference patch before applying the same Otsu cutoff — this is
  what compensates for gray/white-matter intensity differences across the slide.
- Saves the full binary mask, and also **retiles both the image and mask into
  10000×10000 px patches** (`imagesTr` / `labelsTr`) for nnU-Net.
- Finally, **`postprocess_segmentations.py`** filters out noise and processes
  that don't connect to any cell body (`skimage.morphology.remove_small_objects`
  / `remove_small_holes`), then re-tiles the cleaned mask into `labelsTr` as well.
  This cleaned version is what should actually go into nnU-Net's training set.

This local-block thresholding is the source of the tiling artifact discussed later:
each 10000×10000 nnU-Net training/inference tile inherits systematically different
background brightness depending on which 50×50 blocks happen to fall inside it.

Run both, in order:
```bash
sbatch sbatch_microglia_thresholding.sh
# once that's finished:
python postprocess_segmentations.py
```

---

## 2. Manually correct the intensity-thresholded labels

Before these thresholded masks become nnU-Net's training ground truth, review and
hand-correct them. Intensity thresholding is a reasonable starting point but not a
substitute for expert review.

We mostly cleaned the boundaries and linings here. We have found napari to be a helpful tool to clean the segmentations.

---

## 3. Retile for nnU-Net (if starting from a full-slide image + mask, not already tiled)

**Script:** `retile_seg_and_images.py`

Same 10000×10000 non-overlapping tiling logic as step 1, but standalone: used when
you already have a full-resolution image + full-resolution segmentation mask and
just need `imagesTr`/`labelsTr` tile pairs (e.g. re-tiling after correcting a
full-slide mask rather than a single tile).

---

## 4. Set up the nnU-Net raw dataset

Create `nnUNet_raw/Dataset001_BinaryTask/` with `imagesTr/`, `labelsTr/`, and a
`dataset.json`. For this binary microglia task it looked like:
```json
{
  "channel_names": {"0": "Grayscale"},
  "labels": {"background": 0, "target_object": 1},
  "numTraining": 15, # your total number of training images 
  "file_ending": ".tiff",
  "overwrite_image_reader_writer": "NaturalImage2DIO"
}
```
`numTraining` must match the actual file count in `imagesTr`/`labelsTr`.

---

## 5. Plan and preprocess
Create a folder called `model001` in folder `scripts`  

**Script:** `run_preprocess.sh`
```bash
nnUNetv2_plan_and_preprocess -d 1 -c 2d
```
(edit the `-d` flag and `#SBATCH --output=model001/preprocess_output.log` to match
dataset 1 before submitting

---

## 6. Train (5-fold cross-validation)

**Script:** `run_full_train.sh`
```bash
nnUNetv2_train 1 2d ${SLURM_ARRAY_TASK_ID} -tr nnUNetTrainer_250epochs --npz
```
Submitted as a Slurm array job so each fold trains in parallel:
```bash
sbatch --array=0-4 run_full_train.sh
```
(again, update the hardcoded `-d 1`, `--array`, and `--output=model001/...` in the
script before submitting — it's currently set up for a later dataset). Each fold's
log lands in `model001/nnunet_fold_<N>.log`.

> **What to expect:** finding a good model here usually takes a few tries - we
> trained around 6 dataset/config variants before landing on the best one. Our
> best first-round model used 250 epochs, 5-fold cross-validation, on 20 training
> images made up of **8 Set-1 (2 µm) and 12 Set-2 (0.25 µm)** tiles - the
> high-resolution Set-2 examples contributed disproportionately to teaching the
> model finer microglial process/shape detail and boosted its confidence on that
> structure. This first-round model scored a Mean Dice Coefficient of **~75%** - which is a good start, given the training labels themselves came from imperfect intensity
> thresholding (steps 1-2), not hand-drawn ground truth.

---

## 7. Predict on held-out tiles

**Script:** `run_prediction.sh`
```bash
nnUNetv2_predict -i nnUNet_raw/Dataset001_BinaryTask/imagesTs \
                  -o nnUNet_results/Dataset007_BinaryTask/prediction_on_test \
                  -d 1 -f 0 1 2 3 4 -c 2d \
                  -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
                  --save_probabilities
```
`-f 0 1 2 3 4` ensembles all 5 folds. `--save_probabilities` is what makes the
confidence-based QC in the next step possible.

---

## 8. (Optional) Determine / apply nnU-Net postprocessing

**Scripts:** `determine_postprocessing.sh`, `apply_postprocessing.sh`
```bash
nnUNetv2_find_best_configuration 1 -c 2d -tr nnUNetTrainer_250epochs
nnUNetv2_apply_postprocessing -i .../prediction_on_test -o .../prediction_on_test_pp \
  -pp_pkl_file .../postprocessing.pkl -plans_json .../plans.json -np 4
```
This is nnU-Net's own connected-component postprocessing (distinct from the
morphological cleanup in step 1). It is worth checking whether it changes anything;
if pre/post output match, the model is likely near its ceiling for that config.

---

## 9. Confidence-based QC - picking your best slices

**Scripts:** `compute_single_slice_confidence.py`, `visualize_confidence.py`,
`Confidence_heatmap.ipynb`

For each predicted tile's saved probability map (`.npz`), these compute per-slice
metrics: committed confidence (mean probability where foreground was predicted),
fraction of pixels in the "hesitant" 0.2–0.8 band, and foreground pixel count.
Confidence ≥ 0.8 (vs. the nominal 0.5 decision threshold) was used to find the
**best, most reliable, non-tiling-affected predictions**.
These high-confidence slices are what step 10 draws from.

Run via:
```bash
sbatch run_conf_comparison_ps.sh   # compute_single_slice_confidence.py
sbatch run_conf_level_d4.sh        # visualize_confidence.py
```

---

## 10. Fine-tune on inference slices (incremental learning)

**Why:** Qualitative review of step 7's predictions exposes a tiling artifact -
the model learned local background brightness as a proxy for cell density, so
darker tiles get over-segmented and lighter tiles under-segmented at the
boundaries of the 10000×10000 training/inference tiles. Rather than re-labeling
and retraining on the whole dataset from scratch, a small, targeted round of
fine-tuning fixes it: take the **best, non-tiling-affected predictions** flagged
in step 9 (around 15 worked for us), lightly hand-clean them, and feed them back
in as new, high-quality training examples.

Treat this small set as a new training set and go through
**steps 2–9 again**, this time landing in a new dataset (e.g.
`nnUNet_raw/Dataset002_BinaryTask/`) retiling, writing a fresh `dataset.json`,
plan/preprocess, training 5 folds, predicting, and rerunning the confidence QC to
confirm the tiling artifact is gone. This incremental fine-tune is what eliminated
it for us: Mean Dice jumped from ~75% to **~96%**, and previously over/under-labeled
regions became much more uniform after retraining on these examples.

---

## 11. Predict on whole slides, then downsample to 200 µm

Earlier predictions (step 7) ran on 10000×10000 tiles, which were then stitched
back together with a hard, non-overlapping paste and no blending. Each tile was
also normalized on its own statistics during inference, so the seams between
tiles show up as a faint checkerboard in the reconstructed mask. **Avoid this by predicting on each whole slide directly, instead of pre-tiling for
inference.** nnU-Net already runs sliding-window inference internally (patch size
~1024×1536 for this task) and blends overlapping patches with a Gaussian-weighted
average, which only has visibility within a single input image, so
feeding it the whole, untiled slide lets its own stitching handle the entire
slide rather than only the inside of each externally-cut chunk:

```bash
nnUNetv2_predict -i <path_to_full_whole_slide_images> \
                  -o nnUNet_results/Dataset00X_BinaryTask/prediction_on_full_slide \
                  -d <X> -f 0 1 2 3 4 -c 2d \
                  -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
                  --save_probabilities
```
Same command shape as step 7, just pointed at the full, untiled slides rather than
`imagesTs`'s pre-cut 10000×10000 crops — bump `--mem`/GPU memory in the sbatch
script accordingly, since a whole slide (~28000×34000 px) is far larger than a
single tile.

**Script:** `downsample_segmentations.py` (submit via `sbatch_downsample_seg.sh`)
downsamples each predicted slide to 200 µm resolution (Gaussian-smooth `sigma=25`,
then `cv2.resize` by `2/200`), saving one NIfTI per slice. Since predictions are
now already at full resolution in a single file per slide, the tile-stitching loop
in that script (reassembling a 3×4 tile grid) is no longer needed - only the
smoothing/downsampling steps still apply.

> If your GPU can't fit a whole slide in memory, you'll need to fall back to
> tiling for inference. In that case, use overlapping tiles and blend the
> overlap region (linear/cosine ramp, or a Gaussian weight like nnU-Net's own
> importance map) when stitching, rather than the hard paste used previously.

---

## 12. Assemble the 3D volume

**Notebook:** `downsample_nii_rearrange.ipynb`

Loads each per-slice downsampled NIfTI (in slide order, offset by `+10` to center
170 slices in a 191-slice volume) and stacks them into one 3D density map aligned
to the `MicrogliaBlock200um_no-optical` reference volume.

---

## Results

The reconstructed whole-brain density map shows microglial density fluctuating
meaningfully across different parts of the brain rather than sitting flat. A few
patterns stand out:
- Prominent clusters and local variations within cortical layers, highlighting
  region-specific microglial density and organization.
- Clear structural boundaries and density transitions, showing how microglial
  populations map along specific anatomical trajectories rather than being
  uniformly distributed.

## End goal

Downsampled, artifact-corrected microglia density maps reconstructed into a full
3D brain volume, intended for release as an open-source tool for researchers.

---

## Future directions

- **Instance-level segmentation, not just binary masks.** The current target is
  background-vs-microglia. Since morphology encodes functional state (see
  Background), a natural next step is per-cell instance segmentation to actually quantify functional state rather
  than just density.
- **Reduce the manual-correction burden.** Steps 2 and 10 both depend on hand
  correction in napari. Active-learning-style loops (having the model itself
  flag the slices most worth correcting, rather than relying on confidence
  thresholds alone) could shrink how much manual work each fine-tuning round needs.
- **True whole-slide inference at scale.** Step 11 predicts on whole, untiled
  slides where GPU memory allows; formalizing the overlap+blend fallback for
  memory-constrained cases (rather than leaving it as a note) would make the
  pipeline robust on smaller GPUs without reintroducing the tiling artifact.
- **More high-resolution (0.25 µm) coverage.** Set 2 currently covers only 25
  slices of particular glia clusters. Expanding high-res coverage to more
  regions could further improve the model's handling of fine microglial
  morphology beyond what those 25 slices capture.
- **Generalize beyond this one healthy specimen.** The dataset is a single
  healthy Macaque brain. Validating (and likely fine-tuning) on additional
  brains — and eventually diseased/pathological tissue, where microglial
  activation state is often the actual question of interest — would be needed
  before this generalizes as a tool.
- **Public release.** Packaging the trained weights, environment, and pipeline
  (e.g. a container image) for other labs to run directly, rather than needing
  to reproduce training from scratch via this tutorial.

---

## Citation

This project doesn't have a formal publication yet. If you use this pipeline,
please contact **Bradley Karat** (Bradley.Karat@nyulangone.org) or
**Erika Raven** (Erika.Raven@nyulangone.org) for the citation to use.

This pipeline is built directly on nnU-Net — please also cite:
> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H.
> (2021). nnU-Net: a self-configuring method for deep learning-based biomedical
> image segmentation. *Nature Methods*, 18(2), 203–211.

### Background reading

Microglia biology papers referenced in this README:
- Nimmerjahn, A., Kirchhoff, F., & Helmchen, F. (2005). Resting microglial cells
  are highly dynamic surveillants of brain parenchyma in vivo. *Science*,
  308(5726), 1314–1318.
- Paolicelli, R. C., Bolasco, G., Pagani, F., Maggi, L., Scianni, M., Panzanelli,
  P., et al. (2011). Synaptic pruning by microglia is necessary for normal brain
  development. *Science*, 333(6048), 1456–1458.
- Lawson, L. J., Perry, V. H., Dri, P., & Gordon, S. (1990). Heterogeneity in the
  distribution and morphology of microglia in the normal adult mouse brain.
  *Neuroscience*, 39(1), 151–170.
- Norris, G. T., & Kipnis, J. (2018). Immune cells and CNS physiology: Microglia
  and beyond. *Journal of Experimental Medicine*, 216(1), 60–70.
  https://doi.org/10.1084/jem.20180199
- Sousa, C., Biber, K., & Michelucci, A. (2017). Cellular and molecular
  characterization of microglia: A unique immune cell population. *Frontiers in
  Immunology*, 8. https://doi.org/10.3389/fimmu.2017.00198
- Dadwal, S., & Heneka, M. T. (2023). Microglia heterogeneity in health and
  disease. *FEBS Open Bio*, 14(2), 217–229.
  https://doi.org/10.1002/2211-5463.13735
- Hsu, C.-H., Hsu, Y.-Y., Chang, B.-M., Raffensperger, K., Kadden, M., Ton, H. T.,
  Ette, E.-A., Lin, S., Brooks, J., Burke, M. W., Lee, Y.-J., Wang, P. C.,
  Shoykhet, M., & Tu, T.-W. (2025). StainAI: quantitative mapping of stained
  microglia and insights into brain-wide neuroinflammation and therapeutic
  effects in cardiac arrest. *Communications Biology*, 8, 462.
  https://doi.org/10.1038/s42003-025-07926-y
