mkdir /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_swap

cd /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder

for file in *.nii; do
#    if [ ! -f /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder/"$file" ]; then
#        #mincresample "$file" /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled/"$file" -step 0.2 0.2 0.437427 -nelements 340 281 1 -tricubic
#        #mincreshape -dimorder xspace,zspace,yspace /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_resampled/"$file" /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_reorder/"$file"
#    fi
    mrconvert "$file" -axes 0,2,1 /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_swap/"$file"
done

fslmerge -z /gpfs/data/ravenlab/micmac/downsampled_image_test/microglia_nii_concat.nii.gz /gpfs/data/ravenlab/micmac/downsampled_image_test/temp_slices_swap/*.nii
