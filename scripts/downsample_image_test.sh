mkdir /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled
mkdir /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder

cd /gpfs/data/ravenlab/micmac/microglia_2um_nissl_aligned_images

for file in *.mnc; do
#    if [ ! -f /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder/"$file" ]; then
#        #mincresample "$file" /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled/"$file" -step 0.2 0.2 0.437427 -nelements 340 281 1 -tricubic
#        #mincreshape -dimorder xspace,zspace,yspace /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled/"$file" /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder/"$file"
#    fi
    
    mnc2nii /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled/"$file"
done






#mincconcat -concat_dimension yspace /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder/*.mnc /gpfs/data/ravenlab/micmac/downsampled_image_test/microglia_200um_concat.mnc

#mkdir /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_removez

#for file in *.mnc; do
#    mincreshape -dimrange zspace=0,0 /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled/"$file" /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_removez/"$file"
#done

#mincconcat -concat_dimension zspace -start -38.454 -step 0.437427 temp_slices_removez/*.mnc microglia_200um_concat.mnc -clobber

#ls -v /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_removez/*.mnc | tac | xargs mincconcat -concat_dimension zspace -start -38.454 -step 0.437427  -clobber /gpfs/data/ravenlab/micmac/downsampled_image_test/microglia_200um_concat.mnc

#mincreshape -dimorder yspace,zspace,xspace /gpfs/data/ravenlab/micmac/downsampled_image_test/microglia_200um_concat.mnc /gpfs/data/ravenlab/micmac/downsampled_image_test/microglia_200um_concat_reorder.mnc -clobber







#import h5py
#import numpy as np
#
#with h5py.File('microglia_200um_concat_reorder.mnc', 'r+') as f:
#    # Read the scalar values
#    img_max_val = float(f['minc-2.0/image/0/image-max'][()])
#    img_min_val = float(f['minc-2.0/image/0/image-min'][()])
#    print(f"Current min: {img_min_val}, max: {img_max_val}")
#    n_slices = 281  # zspace length
#    # Delete old scalar datasets and replace with 1D arrays
#    del f['minc-2.0/image/0/image-max']
#    del f['minc-2.0/image/0/image-min']
#    ds_max = f.create_dataset('minc-2.0/image/0/image-max',
#                               data=np.full(n_slices, img_max_val, dtype=np.float64))
#    ds_min = f.create_dataset('minc-2.0/image/0/image-min',
#                               data=np.full(n_slices, img_min_val, dtype=np.float64))
#    # Restore attributes
#    for ds, name in [(ds_max, 'image-max'), (ds_min, 'image-min')]:
#        ds.attrs['dimorder'] = np.bytes_('zspace')
#        ds.attrs['varid'] = np.bytes_('MINC standard variable')
#        ds.attrs['vartype'] = np.bytes_('var_attribute')
#        ds.attrs['version'] = np.bytes_('MINC Version    1.0')




#
#import h5py
#import numpy as np
#
#with h5py.File('microglia_200um_concat_reorder.mnc', 'r+') as f:
#    # 1. Swap the dimension groups
#    f.move('minc-2.0/dimensions/yspace', 'minc-2.0/dimensions/zzz_temp')
#    f.move('minc-2.0/dimensions/zspace', 'minc-2.0/dimensions/yspace')
#    f.move('minc-2.0/dimensions/zzz_temp', 'minc-2.0/dimensions/zspace')
#    # 2. Fix the image dimorder
#    img = f['minc-2.0/image/0/image']
#    img.attrs['dimorder'] = np.bytes_('zspace,yspace,xspace')
#    # 3. Fix image-max and image-min dimorders
#    f['minc-2.0/image/0/image-max'].attrs['dimorder'] = np.bytes_('yspace')
#    f['minc-2.0/image/0/image-min'].attrs['dimorder'] = np.bytes_('yspace')
#    # 4. Fix the stale dimorder attr on what is now zspace (was yspace)
#    f['minc-2.0/dimensions/zspace'].attrs['dimorder'] = np.bytes_('zspace')



