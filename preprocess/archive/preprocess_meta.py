from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import SimpleITK as sitk
from dltk.io.preprocessing import whitening
import pandas as pd
import numpy as np
from PIL import Image
import scipy.misc as sci
from operator import itemgetter
from numpy import array, newaxis, expand_dims
import h5py
from dltk.io.augmentation import flip
from dltk.io.preprocessing import whitening
from dltk.io.preprocessing import resize_image_with_crop_or_pad
from patch import extract_class_balanced_example_array


'''
 3D Binary Classification
 Preprocess: Consolidate Optimized Patch-Based Dataset

 Update: 03/08/2019
 Contributors: as1044
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Edema
     - Atelectasis
     - Pneumonia
     - Nodules
'''

PATCH = 112


# Normal


img_path       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Normal/'
fm_hdf         =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Normal.h5', 'r')
features_hdf   =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Normal.h5', 'w')

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:

    img_fn     = str(f)
    subject_id = img_fn.split('.nii.gz')[0].split('CT_')[1]
    binary_mask_fn  = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Normal/_' + subject_id +'_niftynet_out.nii.gz'
    fm_fn      = 'CT_' + subject_id


    # Image
    img        = sitk.ReadImage(img_fn, sitk.sitkFloat32)       # Loading NIFTI Images via SimpleITK
    img        = sitk.GetArrayFromImage(img)
    img        = whitening(img)                                 # Whitening Transformation (Variance=1)
    
    # Binary Mask
    binary_mask     = sitk.GetArrayFromImage(sitk.ReadImage(binary_mask_fn)).astype(np.int32)
    binary_mask     = ((binary_mask == 3)|(binary_mask == 2)).astype(np.int32)

    # Aggregated Feature Maps
    feature_maps = np.array(fm_hdf.get(fm_fn))                                                                       

    
    # Padding Images
    patch_size = [PATCH, PATCH, PATCH]
    img_shape  = img.shape
    
    # Padding Images --Z Dimension
    if (patch_size[0] >=img_shape[0]):
         zdim = patch_size[0]+10
    else:
         zdim = img_shape[0]
    # Padding Images --X Dimension
    if (patch_size[1] >=img_shape[1]):
         xdim = patch_size[1]+10
    else:
         xdim = img_shape[1]
    # Padding Images --Y Dimension
    if (patch_size[2] >=img_shape[2]):
         ydim = patch_size[2]+10
    else:
         ydim = img_shape[2]
    
    img            = resize_image_with_crop_or_pad(img,           [zdim,xdim,ydim], mode='symmetric')
    binary_mask    = resize_image_with_crop_or_pad(binary_mask,   [zdim,xdim,ydim], mode='symmetric')
    feature_maps   = resize_image_with_crop_or_pad(feature_maps,  [zdim,xdim,ydim], mode='symmetric')
    
    # Expand to 4D
    img          = np.expand_dims(img, axis=3)
    feature_maps = np.expand_dims(feature_maps, axis=3)
   
    # Aggregated Input
    img = np.concatenate((img,feature_maps),axis=3)
    
    # Patch Extraction
    img, binary_mask = extract_class_balanced_example_array(
            img, binary_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    = 2,
            classes       = 1)   

    features_hdf.create_dataset(fm_fn, data=img)
    print(fm_fn)

fm_hdf.close()
features_hdf.close()




















# Edema


img_path       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Edema/'
fm_hdf         =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Edema.h5', 'r')
features_hdf   =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Edema.h5', 'w')

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:

    img_fn     = str(f)
    subject_id = img_fn.split('.nii.gz')[0].split('CT_')[1]
    binary_mask_fn  = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/edema/_' + subject_id +'_niftynet_out.nii.gz'
    fm_fn      = 'CT_' + subject_id


    # Image
    img        = sitk.ReadImage(img_fn, sitk.sitkFloat32)       # Loading NIFTI Images via SimpleITK
    img        = sitk.GetArrayFromImage(img)
    img        = whitening(img)                                 # Whitening Transformation (Variance=1)
    
    # Binary Mask
    binary_mask     = sitk.GetArrayFromImage(sitk.ReadImage(binary_mask_fn)).astype(np.int32)
    binary_mask     = ((binary_mask == 3)|(binary_mask == 2)).astype(np.int32)

    # Aggregated Feature Maps
    feature_maps = np.array(fm_hdf.get(fm_fn))                                                                       

    
    # Padding Images
    patch_size = [PATCH, PATCH, PATCH]
    img_shape  = img.shape
    
    # Padding Images --Z Dimension
    if (patch_size[0] >=img_shape[0]):
         zdim = patch_size[0]+10
    else:
         zdim = img_shape[0]
    # Padding Images --X Dimension
    if (patch_size[1] >=img_shape[1]):
         xdim = patch_size[1]+10
    else:
         xdim = img_shape[1]
    # Padding Images --Y Dimension
    if (patch_size[2] >=img_shape[2]):
         ydim = patch_size[2]+10
    else:
         ydim = img_shape[2]
    
    img            = resize_image_with_crop_or_pad(img,           [zdim,xdim,ydim], mode='symmetric')
    binary_mask    = resize_image_with_crop_or_pad(binary_mask,   [zdim,xdim,ydim], mode='symmetric')
    feature_maps   = resize_image_with_crop_or_pad(feature_maps,  [zdim,xdim,ydim], mode='symmetric')
    
    # Expand to 4D
    img          = np.expand_dims(img, axis=3)
    feature_maps = np.expand_dims(feature_maps, axis=3)
   
    # Aggregated Input
    img = np.concatenate((img,feature_maps),axis=3)
    
    # Patch Extraction
    img, binary_mask = extract_class_balanced_example_array(
            img, binary_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    = 2,
            classes       = 1)   

    features_hdf.create_dataset(fm_fn, data=img)
    print(fm_fn)

fm_hdf.close()
features_hdf.close()


















# Pneumonia


img_path       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Pneumonia/'
fm_hdf         =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Pneumonia.h5', 'r')
features_hdf   =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Pneumonia.h5', 'w')

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:

    img_fn     = str(f)
    subject_id = img_fn.split('.nii.gz')[0].split('CT_')[1]
    binary_mask_fn  = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Pneumonia/_' + subject_id +'_niftynet_out.nii.gz'
    fm_fn      = 'CT_' + subject_id


    # Image
    img        = sitk.ReadImage(img_fn, sitk.sitkFloat32)       # Loading NIFTI Images via SimpleITK
    img        = sitk.GetArrayFromImage(img)
    img        = whitening(img)                                 # Whitening Transformation (Variance=1)
    
    # Binary Mask
    binary_mask     = sitk.GetArrayFromImage(sitk.ReadImage(binary_mask_fn)).astype(np.int32)
    binary_mask     = ((binary_mask == 3)|(binary_mask == 2)).astype(np.int32)

    # Aggregated Feature Maps
    feature_maps = np.array(fm_hdf.get(fm_fn))                                                                       

    
    # Padding Images
    patch_size = [PATCH, PATCH, PATCH]
    img_shape  = img.shape
    
    # Padding Images --Z Dimension
    if (patch_size[0] >=img_shape[0]):
         zdim = patch_size[0]+10
    else:
         zdim = img_shape[0]
    # Padding Images --X Dimension
    if (patch_size[1] >=img_shape[1]):
         xdim = patch_size[1]+10
    else:
         xdim = img_shape[1]
    # Padding Images --Y Dimension
    if (patch_size[2] >=img_shape[2]):
         ydim = patch_size[2]+10
    else:
         ydim = img_shape[2]
    
    img            = resize_image_with_crop_or_pad(img,           [zdim,xdim,ydim], mode='symmetric')
    binary_mask    = resize_image_with_crop_or_pad(binary_mask,   [zdim,xdim,ydim], mode='symmetric')
    feature_maps   = resize_image_with_crop_or_pad(feature_maps,  [zdim,xdim,ydim], mode='symmetric')
    
    # Expand to 4D
    img          = np.expand_dims(img, axis=3)
    feature_maps = np.expand_dims(feature_maps, axis=3)
   
    # Aggregated Input
    img = np.concatenate((img,feature_maps),axis=3)
    
    # Patch Extraction
    img, binary_mask = extract_class_balanced_example_array(
            img, binary_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    = 2,
            classes       = 1)   

    features_hdf.create_dataset(fm_fn, data=img)
    print(fm_fn)

fm_hdf.close()
features_hdf.close()

















# Nodules


img_path       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Nodules/'
fm_hdf         =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Nodules.h5', 'r')
features_hdf   =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Nodules.h5', 'w')

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:

    img_fn     = str(f)
    subject_id = img_fn.split('.nii.gz')[0].split('CT_')[1]
    binary_mask_fn  = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Nodules/_' + subject_id +'_niftynet_out.nii.gz'
    fm_fn      = 'CT_' + subject_id


    # Image
    img        = sitk.ReadImage(img_fn, sitk.sitkFloat32)       # Loading NIFTI Images via SimpleITK
    img        = sitk.GetArrayFromImage(img)
    img        = whitening(img)                                 # Whitening Transformation (Variance=1)
    
    # Binary Mask
    binary_mask     = sitk.GetArrayFromImage(sitk.ReadImage(binary_mask_fn)).astype(np.int32)
    binary_mask     = ((binary_mask == 3)|(binary_mask == 2)).astype(np.int32)

    # Aggregated Feature Maps
    feature_maps = np.array(fm_hdf.get(fm_fn))                                                                       

    
    # Padding Images
    patch_size = [PATCH, PATCH, PATCH]
    img_shape  = img.shape
    
    # Padding Images --Z Dimension
    if (patch_size[0] >=img_shape[0]):
         zdim = patch_size[0]+10
    else:
         zdim = img_shape[0]
    # Padding Images --X Dimension
    if (patch_size[1] >=img_shape[1]):
         xdim = patch_size[1]+10
    else:
         xdim = img_shape[1]
    # Padding Images --Y Dimension
    if (patch_size[2] >=img_shape[2]):
         ydim = patch_size[2]+10
    else:
         ydim = img_shape[2]
    
    img            = resize_image_with_crop_or_pad(img,           [zdim,xdim,ydim], mode='symmetric')
    binary_mask    = resize_image_with_crop_or_pad(binary_mask,   [zdim,xdim,ydim], mode='symmetric')
    feature_maps   = resize_image_with_crop_or_pad(feature_maps,  [zdim,xdim,ydim], mode='symmetric')
    
    # Expand to 4D
    img          = np.expand_dims(img, axis=3)
    feature_maps = np.expand_dims(feature_maps, axis=3)
   
    # Aggregated Input
    img = np.concatenate((img,feature_maps),axis=3)
    
    # Patch Extraction
    img, binary_mask = extract_class_balanced_example_array(
            img, binary_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    = 2,
            classes       = 1)   

    features_hdf.create_dataset(fm_fn, data=img)
    print(fm_fn)

fm_hdf.close()
features_hdf.close()






















# Atelectasis


img_path       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Atlectasis/'
fm_hdf         =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Atelectasis.h5', 'r')
features_hdf   =  h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Atelectasis.h5', 'w')

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:

    img_fn     = str(f)
    subject_id = img_fn.split('.nii.gz')[0].split('CT_')[1]
    binary_mask_fn  = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Atelectasis/_' + subject_id +'_niftynet_out.nii.gz'
    fm_fn      = 'CT_' + subject_id


    # Image
    img        = sitk.ReadImage(img_fn, sitk.sitkFloat32)       # Loading NIFTI Images via SimpleITK
    img        = sitk.GetArrayFromImage(img)
    img        = whitening(img)                                 # Whitening Transformation (Variance=1)
    
    # Binary Mask
    binary_mask     = sitk.GetArrayFromImage(sitk.ReadImage(binary_mask_fn)).astype(np.int32)
    binary_mask     = ((binary_mask == 3)|(binary_mask == 2)).astype(np.int32)

    # Aggregated Feature Maps
    feature_maps = np.array(fm_hdf.get(fm_fn))                                                                       

    
    # Padding Images
    patch_size = [PATCH, PATCH, PATCH]
    img_shape  = img.shape
    
    # Padding Images --Z Dimension
    if (patch_size[0] >=img_shape[0]):
         zdim = patch_size[0]+10
    else:
         zdim = img_shape[0]
    # Padding Images --X Dimension
    if (patch_size[1] >=img_shape[1]):
         xdim = patch_size[1]+10
    else:
         xdim = img_shape[1]
    # Padding Images --Y Dimension
    if (patch_size[2] >=img_shape[2]):
         ydim = patch_size[2]+10
    else:
         ydim = img_shape[2]
    
    img            = resize_image_with_crop_or_pad(img,           [zdim,xdim,ydim], mode='symmetric')
    binary_mask    = resize_image_with_crop_or_pad(binary_mask,   [zdim,xdim,ydim], mode='symmetric')
    feature_maps   = resize_image_with_crop_or_pad(feature_maps,  [zdim,xdim,ydim], mode='symmetric')
    
    # Expand to 4D
    img          = np.expand_dims(img, axis=3)
    feature_maps = np.expand_dims(feature_maps, axis=3)
   
    # Aggregated Input
    img = np.concatenate((img,feature_maps),axis=3)
    
    # Patch Extraction
    img, binary_mask = extract_class_balanced_example_array(
            img, binary_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    = 2,
            classes       = 1)   

    features_hdf.create_dataset(fm_fn, data=img)
    print(fm_fn)

fm_hdf.close()
features_hdf.close()