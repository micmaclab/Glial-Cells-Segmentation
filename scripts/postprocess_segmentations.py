import numpy as np
from skimage import morphology, measure, segmentation, filters
import cv2
import tifffile
from tifffile import TiffWriter

def postprocess_microglia(
    binary: np.ndarray,
    min_cell_area: int   = 3,    # px² — tune to your smallest soma
    hole_area: int       = 2,    # px² — max hole to fill inside a cell
    closing_radius: int  = 2,      # px — set 0 to skip closing
    min_branch_len: int  = 10,     # px — prune skeleton branches shorter than this
    remove_border: bool  = True,
    filter_shape: bool   = True,
    min_solidity: float  = None,   # None = don't filter by solidity
    max_solidity: float  = 0.95,   # remove very compact blobs (likely debris)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full post-processing pipeline for microglia binary masks.

    Returns
    -------
    binary_clean : (H, W) bool
    instance_map : (H, W) uint16  — labelled instances
    """
    mask = binary.astype(bool)

    # ── 1. Remove small isolated objects ────────────────────────────────────
    mask = morphology.remove_small_objects(mask, min_size=min_cell_area)

    # ── 2. Fill holes inside cell bodies ────────────────────────────────────
    mask = morphology.remove_small_holes(mask, area_threshold=hole_area)

    # ── 3. Optional closing to reconnect broken processes ───────────────────
    #if closing_radius > 0:
    #    selem = morphology.disk(closing_radius)
    #    mask = morphology.binary_closing(mask, selem)
    #    # Re-run small object removal after closing (can create new small regions)
    #    mask = morphology.remove_small_objects(mask, min_size=min_cell_area)

    # ── 4. Skeleton-based spur pruning ──────────────────────────────────────
    #if min_branch_len > 0:
    #    mask = prune_skeleton_spurs(mask, min_branch_len)

    # ── 5. Remove border-touching instances ─────────────────────────────────
    #if remove_border:
    #    mask = segmentation.clear_border(mask)

    # ── 6. Label instances ──────────────────────────────────────────────────
    labels = measure.label(mask)

    # ── 7. Shape-based filtering ────────────────────────────────────────────
    #if filter_shape:
    #    labels = filter_by_shape(
    #        labels,
    #        min_area=min_cell_area,
    #        max_solidity=max_solidity,
    #    )

    binary_clean = labels > 0
    return binary_clean, labels.astype(np.uint16)


def prune_skeleton_spurs(binary: np.ndarray, min_len: int) -> np.ndarray:
    """
    Skeletonise the mask, remove branches shorter than min_len,
    then reconstruct via dilation back to the original mask.
    Keeps long processes (real microglia branches) and drops
    short disconnected spurs.
    """
    skeleton = morphology.skeletonize(binary)

    # Label connected components of the skeleton
    skel_labels = measure.label(skeleton)
    pruned_skel = np.zeros_like(skeleton)

    for prop in measure.regionprops(skel_labels):
        if prop.area >= min_len:           # skeleton length ≈ pixel count
            pruned_skel[skel_labels == prop.label] = True

    # Reconstruct: keep only original mask pixels that overlap the pruned skeleton
    # Dilate skeleton back to approximate original width
    selem   = morphology.disk(3)
    dilated = morphology.binary_dilation(pruned_skel, selem)
    return binary & dilated


def filter_by_shape(
    labels: np.ndarray,
    min_area: int = 200,
    max_solidity: float = 0.95,
    min_solidity: float = None,
) -> np.ndarray:
    """
    Remove labelled instances that look like debris rather than microglia.

    Debris characteristics:
      - Very small area
      - Very high solidity (compact blob, not branchy)
    
    Microglia characteristics:
      - Larger area
      - Lower solidity (branchy, irregular outline)
      - Higher eccentricity (elongated)
    """
    filtered = np.zeros_like(labels)

    for prop in measure.regionprops(labels):
        if prop.area < min_area:
            continue
        if max_solidity and prop.solidity > max_solidity:
            continue           # too compact — likely debris or nucleus fragment
        if min_solidity and prop.solidity < min_solidity:
            continue           # too fragmented — likely noise
        filtered[labels == prop.label] = prop.label

    return segmentation.relabel_sequential(filtered)[0]


## ── Run ─────────────────────────────────────────────────────────────────────
#if __name__ == "__main__":
#    raw = tifffile.imread("binary_mask.tif").astype(bool)
#
#    binary_clean, instances = postprocess_microglia(
#        raw,
#        min_cell_area  = 200,
#        hole_area      = 100,
#        closing_radius = 2,
#        min_branch_len = 10,
#        remove_border  = True,
#        filter_shape   = True,
#    )
#
#    tifffile.imwrite("binary_clean.tif",  (binary_clean * 255).astype(np.uint8))
#    tifffile.imwrite("instances.tif",     instances)
#    print(f"Cells after filtering: {instances.max()}")



in_dir = '/gpfs/data/ravenlab/micmac'
filenames = np.loadtxt(f'{in_dir}/scripts/microglia_filenames.txt',dtype=str)

for aa in range(len(filenames)):

    print(aa)

    seg_image = tifffile.imread(f'{in_dir}/microglia_2um_nissl_aligned_segmentations/{filenames[aa][0:-4]}.tiff')

    post_image, labels = postprocess_microglia(np.squeeze(seg_image),min_cell_area=25,hole_area=25)

    with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_segmentations_postprocessed/{filenames[aa][0:-4]}.tiff', bigtiff=True) as tif:
        tif.write(post_image)


    # For saving smaller slices as tif for segmentation correction and Unet training
    boxsize = [10000,10000]
    binx = int(np.round(seg_image.shape[0] / boxsize[0]))
    biny = int(np.round(seg_image.shape[1] / boxsize[1]))

    for ii in range(binx):
        for jj in range(biny):
        
            startx = int(ii*boxsize[0])
            endx = int((ii+1) * boxsize[0])
            starty = int(jj*boxsize[1])
            endy = int((jj+1) * boxsize[1])

            seg_hold = post_image[startx:endx,starty:endy]

            with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_segmentations_postprocessed/labelsTr/{filenames[aa][0:-4]}_slice_{ii}_{jj}.tiff', bigtiff=True) as tif:
                tif.write(seg_hold)
