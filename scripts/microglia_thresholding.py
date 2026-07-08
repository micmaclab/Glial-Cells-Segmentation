import numpy as np
from pathlib import Path
#from tqdm import trange
#import matplotlib.pyplot as plt
#from natsort import natsorted
#import dask.array as da
#import dask_image.imread
import cv2
from tifffile import TiffWriter
import nibabel as nib
import skimage.transform
import os
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from skimage.exposure import match_histograms

#MINC already removed colour channels.

#in_dir = '/Users/KARATB01/bigpurple_data/micmac'
in_dir = '/gpfs/data/ravenlab/micmac'

filenames = np.loadtxt(f'{in_dir}/scripts/microglia_filenames.txt',dtype=str)

for aa in range(len(filenames)):

   if not os.path.isfile(f'{in_dir}/microglia_2um_nissl_aligned_segmentations/labelsTr/{filenames[aa][:-4]}_slice_2_2.tiff'):

      print(aa/(len(filenames)))

      img = nib.load(f'{in_dir}/microglia_2um_nissl_aligned_images/{filenames[aa]}')
      affine = img.affine
      hdr = img.header
      img = img.get_fdata()
   
      img = np.squeeze(img[:,:,:])
   
      # GM or low intensity thresholding
      boxsize = [500,500]
      binx = int(np.round(img.shape[0] / boxsize[0])) # 299 x 299 bins
      biny = int(np.round(img.shape[1] / boxsize[1]))
      gray_image = img / np.max(img)
      gray_image = gray_image * 255
      mask = gray_image>15

      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      gray_image_CE = clahe.apply((gray_image.astype(np.uint8)))

      mean_hold = np.empty((binx,biny))

      # Want to find the best N x M box to find threshold on.
      #for ii in range(binx):
      #   for jj in range(biny):
      #   
      #      startx = int(ii*boxsize[0])
      #      endx = int((ii+1) * boxsize[0])
      #      starty = int(jj*boxsize[1])
      #      endy = int((jj+1) * boxsize[1])
   #
      #      img_hold = gray_image[startx:endx,starty:endy]
      #      mask_hold = mask[startx:endx,starty:endy]
   #
      #      mean_hold[ii,jj] = np.mean(img_hold[mask_hold]) / np.sum(mask_hold)
      #      #mean_hold[ii,jj] = np.mean(img_hold[mask_hold]) * np.sum(mask_hold)
   #
   #
      #xind, yind = np.where(mean_hold == np.nanmin(mean_hold))
   #
      #startx = int(xind*boxsize[0])
      #endx = int((xind+1) * boxsize[0])
      #starty = int(yind*boxsize[1])
      #endy = int((yind+1) * boxsize[1])
   #
      #img_hold = gray_image[startx:endx,starty:endy]
      #mask_hold = mask[startx:endx,starty:endy]
   #
      #low_hist = img_hold.flatten()


      # WM or high intensity thresholding
      boxsize = [500,500]
      binx = int(np.round(img.shape[0] / boxsize[0])) # 299 x 299 bins
      biny = int(np.round(img.shape[1] / boxsize[1]))
      mean_hold = np.empty((binx,biny))
      # Want to find the best N x M box to find threshold on.
      for ii in range(binx):
         for jj in range(biny):
         
            startx = int(ii*boxsize[0])
            endx = int((ii+1) * boxsize[0])
            starty = int(jj*boxsize[1])
            endy = int((jj+1) * boxsize[1])
   
            img_hold = gray_image[startx:endx,starty:endy]
            mask_hold = mask[startx:endx,starty:endy]
   
            mean_hold[ii,jj] = np.mean(img_hold[mask_hold]) * np.sum(mask_hold)
   
      xind, yind = np.where(mean_hold == np.nanmax(mean_hold))
   
      startx = int(xind*boxsize[0])
      endx = int((xind+1) * boxsize[0])
      starty = int(yind*boxsize[1])
      endy = int((yind+1) * boxsize[1])
   
      img_hold = gray_image[startx:endx,starty:endy]
      mask_hold = mask[startx:endx,starty:endy]
   
      high_hist = img_hold.flatten()

      otsu_threshold_value_high, binary_image = cv2.threshold(img_hold.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      #seg_image_high = gray_image > (otsu_threshold_value_high-15)

      #img_low_high = np.copy(seg_image_high)
      img_low_high = np.zeros((img.shape))

      #hold = np.logical_and(seg_image_high==0, mask==1)
      #keep_high = np.logical_and(seg_image_high==1, mask==1)

      # Below is because intensity differences in GM/WM and different regions, so trying to match intensity histogram to one used for thresholding
      boxsize = [50,50]
      binx = int(np.round(img.shape[0] / boxsize[0]))
      biny = int(np.round(img.shape[1] / boxsize[1]))
   
      for ii in range(binx):
         for jj in range(biny):
         
            startx = int(ii*boxsize[0])
            endx = int((ii+1) * boxsize[0])
            starty = int(jj*boxsize[1])
            endy = int((jj+1) * boxsize[1])

            if mask[startx:endx,starty:endy].any():

               img_hold = gray_image[startx:endx,starty:endy]
               mask_hold = mask[startx:endx,starty:endy]

               #mean_diff = np.nanmedian(high_hist) - np.nanmedian(img_hold[mask_hold].flatten())
               #img_hold = img_hold + mean_diff
               #img_low_high[startx:endx,starty:endy] = img_hold > otsu_threshold_value_high

               img_hold = match_histograms(img_hold[mask_hold].flatten(), high_hist, channel_axis=None)
               img_hold_2 = np.zeros((boxsize[0],boxsize[1]))
               img_hold_2[mask_hold] = img_hold > otsu_threshold_value_high
               img_low_high[startx:endx,starty:endy] = img_hold_2

      seg_image = img_low_high.astype(bool)

      with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_segmentations/{filenames[aa][:-4]}.tiff', bigtiff=True) as tif:
         tif.write(seg_image)
   
      img_save = nib.Nifti2Image(seg_image.astype(np.uint8), affine)
      nib.save(img_save, f'{in_dir}/microglia_2um_nissl_aligned_segmentations/{filenames[aa][:-4]}.nii.gz')

      with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_images_enhanced/{filenames[aa][:-4]}.tiff', bigtiff=True) as tif:
         tif.write(gray_image_CE)


      # For saving smaller slices as tif for segmentation correction and Unet training
      boxsize = [10000,10000]
      binx = int(np.round(img.shape[0] / boxsize[0]))
      biny = int(np.round(img.shape[1] / boxsize[1]))

      for ii in range(binx):
         for jj in range(biny):
         
            startx = int(ii*boxsize[0])
            endx = int((ii+1) * boxsize[0])
            starty = int(jj*boxsize[1])
            endy = int((jj+1) * boxsize[1])
   
            img_hold = gray_image_CE[startx:endx,starty:endy]
            seg_hold = seg_image[startx:endx,starty:endy]
   
            with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_images_enhanced/imagesTr/{filenames[aa][:-4]}_slice_{ii}_{jj}.tiff', bigtiff=True) as tif: # save as .tif for nnunet training
               tif.write(img_hold.astype(np.uint8))
               
            with TiffWriter(f'{in_dir}/microglia_2um_nissl_aligned_segmentations/labelsTr/{filenames[aa][:-4]}_slice_{ii}_{jj}.tiff', bigtiff=True) as tif:
               tif.write(seg_hold)

