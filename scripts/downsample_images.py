import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import skimage.transform
from scipy import ndimage
import cv2
import os

#MINC already removed colour channels.

in_dir = '/gpfs/data/ravenlab/micmac'

filenames = np.loadtxt(f'{in_dir}/scripts/microglia_filenames.txt',dtype=str)

ref_img = nib.load(f'{in_dir}/histo_volumes/MicrogliaBlock200um_no-optical.mnc')
ref_affine = ref_img.affine
ref_hdr = ref_img.header

arr_num = np.linspace(170,1,170).astype(int) # histo files need to be put into array in reverse order

for aa in range(len(filenames)):

    if os.path.isfile(f'{in_dir}/downsampled_image_test/temp_slices_resampled_nii/41759_G_{arr_num[aa]:03d}.nii.gz'):
        
        seg_down = np.squeeze(nib.load(f'{in_dir}/downsampled_image_test/temp_slices_resampled_nii/41759_G_{arr_num[aa]:03d}.nii.gz').get_fdata()) # start with slice 170

        if aa == 0:
            img_hold = np.empty((seg_down.shape[0],191,seg_down.shape[1]))

    else:

        print(aa)
        #img = nib.load(f'{in_dir}/microglia_2um_nissl_aligned_images/{filenames[aa]}')
        #affine = img.affine
        #hdr = img.header
        affine = nib.load(f'{in_dir}/microglia_2um_nissl_aligned_images/41759_G_{arr_num[aa]:03d}.mnc').affine # start with slice 170
        seg_image = np.squeeze(nib.load(f'{in_dir}/microglia_2um_nissl_aligned_images/41759_G_{arr_num[aa]:03d}.mnc').get_fdata()) # start with slice 170

        seg_image = ndimage.gaussian_filter(seg_image, sigma=10) # smooth

        scale_factor = 2 / 200 
        new_height = int(seg_image.shape[1] * scale_factor)
        new_width = int(seg_image.shape[0] * scale_factor)

        seg_down = cv2.resize(seg_image, (new_height, new_width), interpolation=cv2.INTER_LINEAR)
        seg_down = np.swapaxes(seg_down,0,1)

        if aa == 0:
            img_hold = np.empty((seg_down.shape[0],191,seg_down.shape[1]))

        img_save = nib.Nifti1Image(seg_down, affine)
        nib.save(img_save, f'{in_dir}/downsampled_image_test/temp_slices_resampled_nii/41759_G_{arr_num[aa]:03d}.nii.gz')

    img_hold[:,aa+10,:] = seg_down

#img_swap = np.swapaxes(img_hold,1,2)

img_save = nib.Nifti1Image(img_hold, ref_affine, ref_hdr)
nib.save(img_save, f'{in_dir}/downsampled_image_test/microglia_200um_image_smoothed_downsampled_image.nii.gz')
