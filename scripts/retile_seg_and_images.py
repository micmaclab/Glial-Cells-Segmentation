import numpy as np
from pathlib import Path
#from tqdm import trange
#import matplotlib.pyplot as plt
#from natsort import natsorted
#import dask.array as da
#import dask_image.imread
import cv2
from tifffile import TiffWriter
import tifffile
import nibabel as nib
import skimage.transform
import os
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from skimage.exposure import match_histograms


in_dir = '/gpfs/data/ravenlab/micmac'

filenames = np.loadtxt(f'{in_dir}/scripts/microglia_filenames.txt',dtype=str)

for aa in range(len(filenames)):
    print(aa)

    if not os.path.isfile(f'{in_dir}/microglia_2um_nissl_aligned_images/imagesTr/{filenames[aa][:-4]}_slice_2_3.tiff'):

        img = nib.load(f'{in_dir}/microglia_2um_nissl_aligned_images/{filenames[aa]}')
        affine = img.affine
        hdr = img.header
        img = img.get_fdata()

        img = np.squeeze(img[:,:,:])
        seg_image = tifffile.imread(f'{in_dir}/microglia_2um_nissl_aligned_segmentations/{filenames[aa][:-4]}.tiff')
        #gray_image_CE = tifffile.imread(f'{in_dir}/microglia_2um_nissl_aligned_images_enhanced/{filenames[aa][:-4]}.tiff')

        # For saving smaller slices as tif for segmentation correction and Unet training
        boxsize = [10000,10000]
        binx = int(np.ceil(img.shape[0] / boxsize[0]))
        biny = int(np.ceil(img.shape[1] / boxsize[1]))

        for ii in range(binx):
            for jj in range(biny):
            
                startx = int(ii*boxsize[0])
                endx = int((ii+1) * boxsize[0])
                starty = int(jj*boxsize[1])
                endy = int((jj+1) * boxsize[1])

                #img_CE_hold = gray_image_CE[startx:endx,starty:endy]
                seg_hold = seg_image[startx:endx,starty:endy]
                img_hold = img[startx:endx,starty:endy]

                #with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_images_enhanced/imagesTr/{filenames[aa][:-4]}_slice_{ii}_{jj}.tiff', bigtiff=True) as tif: # save as .tif for nnunet training
                #    tif.write(img_CE_hold.astype(np.uint8))

                with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_segmentations/labelsTr/{filenames[aa][:-4]}_slice_{ii}_{jj}.tiff', bigtiff=True) as tif:
                    tif.write(seg_hold)

                with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_images/imagesTr/{filenames[aa][:-4]}_slice_{ii}_{jj}.tiff', bigtiff=True) as tif:
                    tif.write(img_hold.astype(np.uint8))