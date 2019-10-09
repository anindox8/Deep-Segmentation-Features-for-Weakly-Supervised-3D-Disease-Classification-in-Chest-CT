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
from scipy import ndimage
from sklearn.preprocessing import normalize
from skimage.measure import label


'''
 3D Binary Classification
 Preprocess: Consolidate Optimized Whole Volumes Dataset

 Update: 12/08/2019
 Contributors: as1044
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Emphysema
     - Pneumonia-Atelectasis
     - Mass
     - Nodules
'''

PATCH      = 112
PATCH_NUM  = 2
L_MINAREA  = 3500
B_MINAREA  = 1000
MODE       = 'w'

# Normal

label_name     = 'Normal/'
img_path       = '/DataFolder/lungs/Final_1593/train01/'                        + label_name
bm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/BM_1593/' + label_name
fm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/FM_1593/' + label_name
trainX_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployX/Normal.h5', MODE)
trainY_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployY/Normal.h5', MODE)
trainZ_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployZ/Normal.h5', MODE)


# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Import Paths
    img_fn          = str(f)
    subject_id      = str(img_fn.split('.nii.gz')[0].split('CT_')[1])
    bm_fn           = str(bm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
    fm_fn           = str(fm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')


    # Image
    img             = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
    img             = sitk.GetArrayFromImage(img)
    img             = whitening(img)                                 # Whitening Transformation (Variance=1)

    
    # Binary Masks
    mask            = sitk.GetArrayFromImage(sitk.ReadImage(bm_fn)).astype(np.int32)
    lungs_mask      = ((mask == 3)|(mask == 2)).astype(np.int32)
    body_mask       = ((mask != 0)).astype(np.int32)

    for i in range(0,lungs_mask.shape[0]):                          # Refine Lungs Mask (Remove Holes, Side-Components, Erode Lungs)
        lungs_mask[i,:,:] = ndimage.binary_dilation(lungs_mask[i,:,:]).astype(int)
        lungs_mask[i,:,:] = ndimage.binary_fill_holes(lungs_mask[i,:,:], structure=np.ones((2,2))).astype(int)
        label_im, nb_labels = ndimage.label(lungs_mask[i,:,:])
        sizes = ndimage.sum(lungs_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > L_MINAREA
        lungs_mask[i,:,:] = temp_mask[label_im]
        lungs_mask[i,:,:] = ndimage.binary_erosion(lungs_mask[i,:,:], structure=np.ones((10,10))).astype(int)

    for m in range(lungs_mask.shape[0]):                            # Remove Oversegmentation on Dead Slices
        if (img[m].mean() <= -0.90):
            lungs_mask[m] = 0                

    for i in range(0,body_mask.shape[0]):                           # Refine Full Body Mask (Remove Holes, Side-Components)
        body_mask[i,:,:] = ndimage.binary_dilation(body_mask[i,:,:]).astype(int)
        body_mask[i,:,:] = ndimage.binary_fill_holes(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)
        label_im, nb_labels = ndimage.label(body_mask[i,:,:])
        sizes = ndimage.sum(body_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > B_MINAREA
        body_mask[i,:,:] = temp_mask[label_im]
        body_mask[i,:,:] = ndimage.binary_erosion(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)


    # Feature Maps
    file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(fm_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)                                                                  

    aoo = np.expand_dims(feature_maps[:,:,:,3]*body_mask,axis=3)    # Target 13 Feature Components
    boo = np.expand_dims(feature_maps[:,:,:,6]*body_mask,axis=3)
    coo = np.expand_dims(feature_maps[:,:,:,8]*body_mask,axis=3)
    doo = np.expand_dims(feature_maps[:,:,:,14]*body_mask,axis=3)
    eoo = np.expand_dims(feature_maps[:,:,:,16]*body_mask,axis=3)    
    foo = np.expand_dims(feature_maps[:,:,:,20]*body_mask,axis=3)
    goo = np.expand_dims(feature_maps[:,:,:,22]*body_mask,axis=3)
    hoo = np.expand_dims(feature_maps[:,:,:,25]*body_mask,axis=3)
    ioo = np.expand_dims(feature_maps[:,:,:,35]*body_mask,axis=3)
    joo = np.expand_dims(feature_maps[:,:,:,36]*body_mask,axis=3)
    koo = np.expand_dims(feature_maps[:,:,:,47]*body_mask,axis=3)
    loo = np.expand_dims(feature_maps[:,:,:,54]*body_mask,axis=3)
    moo = np.expand_dims(feature_maps[:,:,:,55]*body_mask,axis=3)

    # Aggregate and Normalize Feature Maps
    aggregated_featuremaps   = np.concatenate((aoo,boo,coo,doo,eoo,foo,goo,hoo,ioo,joo,koo,loo,moo),axis=3)
    aggregated_featuremaps   = np.mean(aggregated_featuremaps,axis=3)
    aggregated_featuremaps   = (aggregated_featuremaps-aggregated_featuremaps.min())/(aggregated_featuremaps.max()-aggregated_featuremaps.min())


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
    
    img                      = resize_image_with_crop_or_pad(img,                    [zdim,xdim,ydim], mode='symmetric')
    lungs_mask               = resize_image_with_crop_or_pad(lungs_mask,             [zdim,xdim,ydim], mode='symmetric')
    aggregated_featuremaps   = resize_image_with_crop_or_pad(aggregated_featuremaps, [zdim,xdim,ydim], mode='symmetric')



    # Alternate Aggregation (Experimental) --------------------------------------------------------------------------------------------------------
    aooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,3]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3)
    booZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,6]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    cooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,8]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    dooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,14]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    eooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,16]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    fooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,20]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    gooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,22]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    hooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,25]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    iooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,35]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    jooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,36]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    kooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,47]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    looZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,54]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    mooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,55]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
       
    imgZ                     = np.expand_dims(img,axis=3)
    aggregated_featuremapsZ  = np.concatenate((aooZ,booZ,cooZ,dooZ,eooZ,fooZ,gooZ,hooZ,iooZ,jooZ,kooZ,looZ,mooZ),axis=3)
    aggregated_featuremapsZ  = (aggregated_featuremapsZ-aggregated_featuremapsZ.min())/(aggregated_featuremapsZ.max()-aggregated_featuremapsZ.min())
    imgZ                     = np.concatenate((imgZ, aggregated_featuremapsZ),axis=3)

    '''
    imgZ, _ = extract_class_balanced_example_array(
            imgZ, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )
    '''      
    # ----------------------------------------------------------------------------------------------------------------------------------------------



    # (3D -> 4D Conversion) -> Aggregated Input Volume
    img                      = np.expand_dims(img,                          axis=3)
    aggregated_featuremaps   = np.expand_dims(aggregated_featuremaps,       axis=3)
    img                      = np.concatenate((img,aggregated_featuremaps), axis=3)
    
    '''
    # Patch Extraction
    img, lungs_mask = extract_class_balanced_example_array(
            img, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )   
    '''

    # Patch Export
    trainX_hdf.create_dataset(subject_id, data=np.expand_dims(img[:,:,:,0], axis=3))     # 1 CT                         (Baseline Method)
    trainY_hdf.create_dataset(subject_id, data=img)                                      # 1 CT + AggregatedFeatureMap  (Standard Method)
    trainZ_hdf.create_dataset(subject_id, data=imgZ)                                     # 1 CT + 13FeatureMaps         (Experimental Method)

    print(label_name + 'CT_' + subject_id)

trainX_hdf.close()
trainY_hdf.close()
trainZ_hdf.close()













# Emphysema

label_name     = 'Emphysema/'
img_path       = '/DataFolder/lungs/Final_1593/train01/'                        + label_name
bm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/BM_1593/' + label_name
fm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/FM_1593/' + label_name
trainX_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployX/Emphysema.h5', MODE)
trainY_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployY/Emphysema.h5', MODE)
trainZ_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployZ/Emphysema.h5', MODE)

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Import Paths
    img_fn          = str(f)
    subject_id      = str(img_fn.split('.nii.gz')[0].split('CT_')[1])
    bm_fn           = str(bm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
    fm_fn           = str(fm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')


    # Image
    img             = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
    img             = sitk.GetArrayFromImage(img)
    img             = whitening(img)                                 # Whitening Transformation (Variance=1)

    
    # Binary Masks
    mask            = sitk.GetArrayFromImage(sitk.ReadImage(bm_fn)).astype(np.int32)
    lungs_mask      = ((mask == 3)|(mask == 2)).astype(np.int32)
    body_mask       = ((mask != 0)).astype(np.int32)

    for i in range(0,lungs_mask.shape[0]):                          # Refine Lungs Mask (Remove Holes, Side-Components, Erode Lungs)
        lungs_mask[i,:,:] = ndimage.binary_dilation(lungs_mask[i,:,:]).astype(int)
        lungs_mask[i,:,:] = ndimage.binary_fill_holes(lungs_mask[i,:,:], structure=np.ones((2,2))).astype(int)
        label_im, nb_labels = ndimage.label(lungs_mask[i,:,:])
        sizes = ndimage.sum(lungs_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > L_MINAREA
        lungs_mask[i,:,:] = temp_mask[label_im]
        lungs_mask[i,:,:] = ndimage.binary_erosion(lungs_mask[i,:,:], structure=np.ones((10,10))).astype(int)

    for m in range(lungs_mask.shape[0]):                            # Remove Oversegmentation on Dead Slices
        if (img[m].mean() <= -0.90):
            lungs_mask[m] = 0                

    for i in range(0,body_mask.shape[0]):                           # Refine Full Body Mask (Remove Holes, Side-Components)
        body_mask[i,:,:] = ndimage.binary_dilation(body_mask[i,:,:]).astype(int)
        body_mask[i,:,:] = ndimage.binary_fill_holes(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)
        label_im, nb_labels = ndimage.label(body_mask[i,:,:])
        sizes = ndimage.sum(body_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > B_MINAREA
        body_mask[i,:,:] = temp_mask[label_im]
        body_mask[i,:,:] = ndimage.binary_erosion(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)


    # Feature Maps
    file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(fm_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)                                                                  

    aoo = np.expand_dims(feature_maps[:,:,:,3]*body_mask,axis=3)    # Target 13 Feature Components
    boo = np.expand_dims(feature_maps[:,:,:,6]*body_mask,axis=3)
    coo = np.expand_dims(feature_maps[:,:,:,8]*body_mask,axis=3)
    doo = np.expand_dims(feature_maps[:,:,:,14]*body_mask,axis=3)
    eoo = np.expand_dims(feature_maps[:,:,:,16]*body_mask,axis=3)    
    foo = np.expand_dims(feature_maps[:,:,:,20]*body_mask,axis=3)
    goo = np.expand_dims(feature_maps[:,:,:,22]*body_mask,axis=3)
    hoo = np.expand_dims(feature_maps[:,:,:,25]*body_mask,axis=3)
    ioo = np.expand_dims(feature_maps[:,:,:,35]*body_mask,axis=3)
    joo = np.expand_dims(feature_maps[:,:,:,36]*body_mask,axis=3)
    koo = np.expand_dims(feature_maps[:,:,:,47]*body_mask,axis=3)
    loo = np.expand_dims(feature_maps[:,:,:,54]*body_mask,axis=3)
    moo = np.expand_dims(feature_maps[:,:,:,55]*body_mask,axis=3)

    # Aggregate and Normalize Feature Maps
    aggregated_featuremaps   = np.concatenate((aoo,boo,coo,doo,eoo,foo,goo,hoo,ioo,joo,koo,loo,moo),axis=3)
    aggregated_featuremaps   = np.mean(aggregated_featuremaps,axis=3)
    aggregated_featuremaps   = (aggregated_featuremaps-aggregated_featuremaps.min())/(aggregated_featuremaps.max()-aggregated_featuremaps.min())


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
    
    img                      = resize_image_with_crop_or_pad(img,                    [zdim,xdim,ydim], mode='symmetric')
    lungs_mask               = resize_image_with_crop_or_pad(lungs_mask,             [zdim,xdim,ydim], mode='symmetric')
    aggregated_featuremaps   = resize_image_with_crop_or_pad(aggregated_featuremaps, [zdim,xdim,ydim], mode='symmetric')



    # Alternate Aggregation (Experimental) --------------------------------------------------------------------------------------------------------
    aooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,3]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3)
    booZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,6]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    cooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,8]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    dooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,14]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    eooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,16]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    fooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,20]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    gooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,22]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    hooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,25]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    iooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,35]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    jooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,36]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    kooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,47]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    looZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,54]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    mooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,55]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
       
    imgZ                     = np.expand_dims(img,axis=3)
    aggregated_featuremapsZ  = np.concatenate((aooZ,booZ,cooZ,dooZ,eooZ,fooZ,gooZ,hooZ,iooZ,jooZ,kooZ,looZ,mooZ),axis=3)
    aggregated_featuremapsZ  = (aggregated_featuremapsZ-aggregated_featuremapsZ.min())/(aggregated_featuremapsZ.max()-aggregated_featuremapsZ.min())
    imgZ                     = np.concatenate((imgZ, aggregated_featuremapsZ),axis=3)

    '''
    imgZ, _ = extract_class_balanced_example_array(
            imgZ, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )
    '''        
    # ----------------------------------------------------------------------------------------------------------------------------------------------



    # (3D -> 4D Conversion) -> Aggregated Input Volume
    img                      = np.expand_dims(img,                          axis=3)
    aggregated_featuremaps   = np.expand_dims(aggregated_featuremaps,       axis=3)
    img                      = np.concatenate((img,aggregated_featuremaps), axis=3)
    
    '''
    # Patch Extraction
    img, lungs_mask = extract_class_balanced_example_array(
            img, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )   
    '''

    # Patch Export
    trainX_hdf.create_dataset(subject_id, data=np.expand_dims(img[:,:,:,0], axis=3))     # 1 CT                         (Baseline Method)
    trainY_hdf.create_dataset(subject_id, data=img)                                      # 1 CT + AggregatedFeatureMap  (Standard Method)
    trainZ_hdf.create_dataset(subject_id, data=imgZ)                                     # 1 CT + 13FeatureMaps         (Experimental Method)

    print(label_name + 'CT_' + subject_id)


trainX_hdf.close()
trainY_hdf.close()
trainZ_hdf.close()














# Nodules

label_name     = 'Nodules/'
img_path       = '/DataFolder/lungs/Final_1593/train01/'                        + label_name
bm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/BM_1593/' + label_name
fm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/FM_1593/' + label_name
trainX_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployX/Nodules.h5', MODE)
trainY_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployY/Nodules.h5', MODE)
trainZ_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployZ/Nodules.h5', MODE)

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Import Paths
    img_fn          = str(f)
    subject_id      = str(img_fn.split('.nii.gz')[0].split('CT_')[1])
    bm_fn           = str(bm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
    fm_fn           = str(fm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')


    # Image
    img             = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
    img             = sitk.GetArrayFromImage(img)
    img             = whitening(img)                                 # Whitening Transformation (Variance=1)

    
    # Binary Masks
    mask            = sitk.GetArrayFromImage(sitk.ReadImage(bm_fn)).astype(np.int32)
    lungs_mask      = ((mask == 3)|(mask == 2)).astype(np.int32)
    body_mask       = ((mask != 0)).astype(np.int32)

    for i in range(0,lungs_mask.shape[0]):                          # Refine Lungs Mask (Remove Holes, Side-Components, Erode Lungs)
        lungs_mask[i,:,:] = ndimage.binary_dilation(lungs_mask[i,:,:]).astype(int)
        lungs_mask[i,:,:] = ndimage.binary_fill_holes(lungs_mask[i,:,:], structure=np.ones((2,2))).astype(int)
        label_im, nb_labels = ndimage.label(lungs_mask[i,:,:])
        sizes = ndimage.sum(lungs_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > L_MINAREA
        lungs_mask[i,:,:] = temp_mask[label_im]
        lungs_mask[i,:,:] = ndimage.binary_erosion(lungs_mask[i,:,:], structure=np.ones((10,10))).astype(int)

    for m in range(lungs_mask.shape[0]):                            # Remove Oversegmentation on Dead Slices
        if (img[m].mean() <= -0.90):
            lungs_mask[m] = 0                

    for i in range(0,body_mask.shape[0]):                           # Refine Full Body Mask (Remove Holes, Side-Components)
        body_mask[i,:,:] = ndimage.binary_dilation(body_mask[i,:,:]).astype(int)
        body_mask[i,:,:] = ndimage.binary_fill_holes(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)
        label_im, nb_labels = ndimage.label(body_mask[i,:,:])
        sizes = ndimage.sum(body_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > B_MINAREA
        body_mask[i,:,:] = temp_mask[label_im]
        body_mask[i,:,:] = ndimage.binary_erosion(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)


    # Feature Maps
    file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(fm_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)                                                                  

    aoo = np.expand_dims(feature_maps[:,:,:,3]*body_mask,axis=3)    # Target 13 Feature Components
    boo = np.expand_dims(feature_maps[:,:,:,6]*body_mask,axis=3)
    coo = np.expand_dims(feature_maps[:,:,:,8]*body_mask,axis=3)
    doo = np.expand_dims(feature_maps[:,:,:,14]*body_mask,axis=3)
    eoo = np.expand_dims(feature_maps[:,:,:,16]*body_mask,axis=3)    
    foo = np.expand_dims(feature_maps[:,:,:,20]*body_mask,axis=3)
    goo = np.expand_dims(feature_maps[:,:,:,22]*body_mask,axis=3)
    hoo = np.expand_dims(feature_maps[:,:,:,25]*body_mask,axis=3)
    ioo = np.expand_dims(feature_maps[:,:,:,35]*body_mask,axis=3)
    joo = np.expand_dims(feature_maps[:,:,:,36]*body_mask,axis=3)
    koo = np.expand_dims(feature_maps[:,:,:,47]*body_mask,axis=3)
    loo = np.expand_dims(feature_maps[:,:,:,54]*body_mask,axis=3)
    moo = np.expand_dims(feature_maps[:,:,:,55]*body_mask,axis=3)

    # Aggregate and Normalize Feature Maps
    aggregated_featuremaps   = np.concatenate((aoo,boo,coo,doo,eoo,foo,goo,hoo,ioo,joo,koo,loo,moo),axis=3)
    aggregated_featuremaps   = np.mean(aggregated_featuremaps,axis=3)
    aggregated_featuremaps   = (aggregated_featuremaps-aggregated_featuremaps.min())/(aggregated_featuremaps.max()-aggregated_featuremaps.min())


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
    
    img                      = resize_image_with_crop_or_pad(img,                    [zdim,xdim,ydim], mode='symmetric')
    lungs_mask               = resize_image_with_crop_or_pad(lungs_mask,             [zdim,xdim,ydim], mode='symmetric')
    aggregated_featuremaps   = resize_image_with_crop_or_pad(aggregated_featuremaps, [zdim,xdim,ydim], mode='symmetric')



    # Alternate Aggregation (Experimental) --------------------------------------------------------------------------------------------------------
    aooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,3]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3)
    booZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,6]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    cooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,8]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    dooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,14]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    eooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,16]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    fooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,20]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    gooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,22]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    hooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,25]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    iooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,35]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    jooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,36]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    kooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,47]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    looZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,54]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    mooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,55]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
       
    imgZ                     = np.expand_dims(img,axis=3)
    aggregated_featuremapsZ  = np.concatenate((aooZ,booZ,cooZ,dooZ,eooZ,fooZ,gooZ,hooZ,iooZ,jooZ,kooZ,looZ,mooZ),axis=3)
    aggregated_featuremapsZ  = (aggregated_featuremapsZ-aggregated_featuremapsZ.min())/(aggregated_featuremapsZ.max()-aggregated_featuremapsZ.min())
    imgZ                     = np.concatenate((imgZ, aggregated_featuremapsZ),axis=3)

    '''
    imgZ, _ = extract_class_balanced_example_array(
            imgZ, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )
    '''
    # ----------------------------------------------------------------------------------------------------------------------------------------------



    # (3D -> 4D Conversion) -> Aggregated Input Volume
    img                      = np.expand_dims(img,                          axis=3)
    aggregated_featuremaps   = np.expand_dims(aggregated_featuremaps,       axis=3)
    img                      = np.concatenate((img,aggregated_featuremaps), axis=3)
    
    '''
    # Patch Extraction
    img, lungs_mask = extract_class_balanced_example_array(
            img, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )   
    '''

    # Patch Export
    trainX_hdf.create_dataset(subject_id, data=np.expand_dims(img[:,:,:,0], axis=3))     # 1 CT                         (Baseline Method)
    trainY_hdf.create_dataset(subject_id, data=img)                                      # 1 CT + AggregatedFeatureMap  (Standard Method)
    trainZ_hdf.create_dataset(subject_id, data=imgZ)                                     # 1 CT + 13FeatureMaps         (Experimental Method)

    print(label_name + 'CT_' + subject_id)

trainX_hdf.close()
trainY_hdf.close()
trainZ_hdf.close()














# Mass

label_name     = 'Mass/'
img_path       = '/DataFolder/lungs/Final_1593/train01/'                        + label_name
bm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/BM_1593/' + label_name
fm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/FM_1593/' + label_name
trainX_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployX/Mass.h5', MODE)
trainY_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployY/Mass.h5', MODE)
trainZ_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployZ/Mass.h5', MODE)

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Import Paths
    img_fn          = str(f)
    subject_id      = str(img_fn.split('.nii.gz')[0].split('CT_')[1])
    bm_fn           = str(bm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
    fm_fn           = str(fm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')


    # Image
    img             = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
    img             = sitk.GetArrayFromImage(img)
    img             = whitening(img)                                 # Whitening Transformation (Variance=1)

    
    # Binary Masks
    mask            = sitk.GetArrayFromImage(sitk.ReadImage(bm_fn)).astype(np.int32)
    lungs_mask      = ((mask == 3)|(mask == 2)).astype(np.int32)
    body_mask       = ((mask != 0)).astype(np.int32)

    for i in range(0,lungs_mask.shape[0]):                          # Refine Lungs Mask (Remove Holes, Side-Components, Erode Lungs)
        lungs_mask[i,:,:] = ndimage.binary_dilation(lungs_mask[i,:,:]).astype(int)
        lungs_mask[i,:,:] = ndimage.binary_fill_holes(lungs_mask[i,:,:], structure=np.ones((2,2))).astype(int)
        label_im, nb_labels = ndimage.label(lungs_mask[i,:,:])
        sizes = ndimage.sum(lungs_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > L_MINAREA
        lungs_mask[i,:,:] = temp_mask[label_im]
        lungs_mask[i,:,:] = ndimage.binary_erosion(lungs_mask[i,:,:], structure=np.ones((10,10))).astype(int)

    for m in range(lungs_mask.shape[0]):                            # Remove Oversegmentation on Dead Slices
        if (img[m].mean() <= -0.90):
            lungs_mask[m] = 0                

    for i in range(0,body_mask.shape[0]):                           # Refine Full Body Mask (Remove Holes, Side-Components)
        body_mask[i,:,:] = ndimage.binary_dilation(body_mask[i,:,:]).astype(int)
        body_mask[i,:,:] = ndimage.binary_fill_holes(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)
        label_im, nb_labels = ndimage.label(body_mask[i,:,:])
        sizes = ndimage.sum(body_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > B_MINAREA
        body_mask[i,:,:] = temp_mask[label_im]
        body_mask[i,:,:] = ndimage.binary_erosion(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)


    # Feature Maps
    file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(fm_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)                                                                  

    aoo = np.expand_dims(feature_maps[:,:,:,3]*body_mask,axis=3)    # Target 13 Feature Components
    boo = np.expand_dims(feature_maps[:,:,:,6]*body_mask,axis=3)
    coo = np.expand_dims(feature_maps[:,:,:,8]*body_mask,axis=3)
    doo = np.expand_dims(feature_maps[:,:,:,14]*body_mask,axis=3)
    eoo = np.expand_dims(feature_maps[:,:,:,16]*body_mask,axis=3)    
    foo = np.expand_dims(feature_maps[:,:,:,20]*body_mask,axis=3)
    goo = np.expand_dims(feature_maps[:,:,:,22]*body_mask,axis=3)
    hoo = np.expand_dims(feature_maps[:,:,:,25]*body_mask,axis=3)
    ioo = np.expand_dims(feature_maps[:,:,:,35]*body_mask,axis=3)
    joo = np.expand_dims(feature_maps[:,:,:,36]*body_mask,axis=3)
    koo = np.expand_dims(feature_maps[:,:,:,47]*body_mask,axis=3)
    loo = np.expand_dims(feature_maps[:,:,:,54]*body_mask,axis=3)
    moo = np.expand_dims(feature_maps[:,:,:,55]*body_mask,axis=3)

    # Aggregate and Normalize Feature Maps
    aggregated_featuremaps   = np.concatenate((aoo,boo,coo,doo,eoo,foo,goo,hoo,ioo,joo,koo,loo,moo),axis=3)
    aggregated_featuremaps   = np.mean(aggregated_featuremaps,axis=3)
    aggregated_featuremaps   = (aggregated_featuremaps-aggregated_featuremaps.min())/(aggregated_featuremaps.max()-aggregated_featuremaps.min())


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
    
    img                      = resize_image_with_crop_or_pad(img,                    [zdim,xdim,ydim], mode='symmetric')
    lungs_mask               = resize_image_with_crop_or_pad(lungs_mask,             [zdim,xdim,ydim], mode='symmetric')
    aggregated_featuremaps   = resize_image_with_crop_or_pad(aggregated_featuremaps, [zdim,xdim,ydim], mode='symmetric')



    # Alternate Aggregation (Experimental) --------------------------------------------------------------------------------------------------------
    aooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,3]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3)
    booZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,6]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    cooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,8]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    dooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,14]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    eooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,16]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    fooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,20]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    gooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,22]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    hooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,25]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    iooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,35]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    jooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,36]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    kooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,47]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    looZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,54]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    mooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,55]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
       
    imgZ                     = np.expand_dims(img,axis=3)
    aggregated_featuremapsZ  = np.concatenate((aooZ,booZ,cooZ,dooZ,eooZ,fooZ,gooZ,hooZ,iooZ,jooZ,kooZ,looZ,mooZ),axis=3)
    aggregated_featuremapsZ  = (aggregated_featuremapsZ-aggregated_featuremapsZ.min())/(aggregated_featuremapsZ.max()-aggregated_featuremapsZ.min())
    imgZ                     = np.concatenate((imgZ, aggregated_featuremapsZ),axis=3)

    '''
    imgZ, _ = extract_class_balanced_example_array(
            imgZ, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )
    '''        
    # ----------------------------------------------------------------------------------------------------------------------------------------------



    # (3D -> 4D Conversion) -> Aggregated Input Volume
    img                      = np.expand_dims(img,                          axis=3)
    aggregated_featuremaps   = np.expand_dims(aggregated_featuremaps,       axis=3)
    img                      = np.concatenate((img,aggregated_featuremaps), axis=3)
    
    '''
    # Patch Extraction
    img, lungs_mask = extract_class_balanced_example_array(
            img, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )   
    '''

    # Patch Export
    trainX_hdf.create_dataset(subject_id, data=np.expand_dims(img[:,:,:,0], axis=3))     # 1 CT                         (Baseline Method)
    trainY_hdf.create_dataset(subject_id, data=img)                                      # 1 CT + AggregatedFeatureMap  (Standard Method)
    trainZ_hdf.create_dataset(subject_id, data=imgZ)                                     # 1 CT + 13FeatureMaps         (Experimental Method)

    print(label_name + 'CT_' + subject_id)

trainX_hdf.close()
trainY_hdf.close()
trainZ_hdf.close()














# Pneumonia-Atelectasis

label_name     = 'Pneumonia-Atelectasis/'
img_path       = '/DataFolder/lungs/Final_1593/train01/'                        + label_name
bm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/BM_1593/' + label_name
fm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/FM_1593/' + label_name
trainX_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployX/Pneumonia-Atelectasis.h5', MODE)
trainY_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployY/Pneumonia-Atelectasis.h5', MODE)
trainZ_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployZ/Pneumonia-Atelectasis.h5', MODE)

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Import Paths
    img_fn          = str(f)
    subject_id      = str(img_fn.split('.nii.gz')[0].split('CT_')[1])
    bm_fn           = str(bm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
    fm_fn           = str(fm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')


    # Image
    img             = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
    img             = sitk.GetArrayFromImage(img)
    img             = whitening(img)                                 # Whitening Transformation (Variance=1)

    
    # Binary Masks
    mask            = sitk.GetArrayFromImage(sitk.ReadImage(bm_fn)).astype(np.int32)
    lungs_mask      = ((mask == 3)|(mask == 2)).astype(np.int32)
    body_mask       = ((mask != 0)).astype(np.int32)

    for i in range(0,lungs_mask.shape[0]):                          # Refine Lungs Mask (Remove Holes, Side-Components, Erode Lungs)
        lungs_mask[i,:,:] = ndimage.binary_dilation(lungs_mask[i,:,:]).astype(int)
        lungs_mask[i,:,:] = ndimage.binary_fill_holes(lungs_mask[i,:,:], structure=np.ones((2,2))).astype(int)
        label_im, nb_labels = ndimage.label(lungs_mask[i,:,:])
        sizes = ndimage.sum(lungs_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > L_MINAREA
        lungs_mask[i,:,:] = temp_mask[label_im]
        lungs_mask[i,:,:] = ndimage.binary_erosion(lungs_mask[i,:,:], structure=np.ones((10,10))).astype(int)

    for m in range(lungs_mask.shape[0]):                            # Remove Oversegmentation on Dead Slices
        if (img[m].mean() <= -0.90):
            lungs_mask[m] = 0                

    for i in range(0,body_mask.shape[0]):                           # Refine Full Body Mask (Remove Holes, Side-Components)
        body_mask[i,:,:] = ndimage.binary_dilation(body_mask[i,:,:]).astype(int)
        body_mask[i,:,:] = ndimage.binary_fill_holes(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)
        label_im, nb_labels = ndimage.label(body_mask[i,:,:])
        sizes = ndimage.sum(body_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > B_MINAREA
        body_mask[i,:,:] = temp_mask[label_im]
        body_mask[i,:,:] = ndimage.binary_erosion(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)


    # Feature Maps
    file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(fm_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)                                                                  

    aoo = np.expand_dims(feature_maps[:,:,:,3]*body_mask,axis=3)    # Target 13 Feature Components
    boo = np.expand_dims(feature_maps[:,:,:,6]*body_mask,axis=3)
    coo = np.expand_dims(feature_maps[:,:,:,8]*body_mask,axis=3)
    doo = np.expand_dims(feature_maps[:,:,:,14]*body_mask,axis=3)
    eoo = np.expand_dims(feature_maps[:,:,:,16]*body_mask,axis=3)    
    foo = np.expand_dims(feature_maps[:,:,:,20]*body_mask,axis=3)
    goo = np.expand_dims(feature_maps[:,:,:,22]*body_mask,axis=3)
    hoo = np.expand_dims(feature_maps[:,:,:,25]*body_mask,axis=3)
    ioo = np.expand_dims(feature_maps[:,:,:,35]*body_mask,axis=3)
    joo = np.expand_dims(feature_maps[:,:,:,36]*body_mask,axis=3)
    koo = np.expand_dims(feature_maps[:,:,:,47]*body_mask,axis=3)
    loo = np.expand_dims(feature_maps[:,:,:,54]*body_mask,axis=3)
    moo = np.expand_dims(feature_maps[:,:,:,55]*body_mask,axis=3)

    # Aggregate and Normalize Feature Maps
    aggregated_featuremaps   = np.concatenate((aoo,boo,coo,doo,eoo,foo,goo,hoo,ioo,joo,koo,loo,moo),axis=3)
    aggregated_featuremaps   = np.mean(aggregated_featuremaps,axis=3)
    aggregated_featuremaps   = (aggregated_featuremaps-aggregated_featuremaps.min())/(aggregated_featuremaps.max()-aggregated_featuremaps.min())


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
    
    img                      = resize_image_with_crop_or_pad(img,                    [zdim,xdim,ydim], mode='symmetric')
    lungs_mask               = resize_image_with_crop_or_pad(lungs_mask,             [zdim,xdim,ydim], mode='symmetric')
    aggregated_featuremaps   = resize_image_with_crop_or_pad(aggregated_featuremaps, [zdim,xdim,ydim], mode='symmetric')



    # Alternate Aggregation (Experimental) --------------------------------------------------------------------------------------------------------
    aooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,3]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3)
    booZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,6]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    cooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,8]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    dooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,14]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    eooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,16]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    fooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,20]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    gooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,22]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    hooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,25]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    iooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,35]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    jooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,36]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    kooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,47]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    looZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,54]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    mooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,55]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
       
    imgZ                     = np.expand_dims(img,axis=3)
    aggregated_featuremapsZ  = np.concatenate((aooZ,booZ,cooZ,dooZ,eooZ,fooZ,gooZ,hooZ,iooZ,jooZ,kooZ,looZ,mooZ),axis=3)
    aggregated_featuremapsZ  = (aggregated_featuremapsZ-aggregated_featuremapsZ.min())/(aggregated_featuremapsZ.max()-aggregated_featuremapsZ.min())
    imgZ                     = np.concatenate((imgZ, aggregated_featuremapsZ),axis=3)

    '''
    imgZ, _ = extract_class_balanced_example_array(
            imgZ, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )
    '''
    # ----------------------------------------------------------------------------------------------------------------------------------------------



    # (3D -> 4D Conversion) -> Aggregated Input Volume
    img                      = np.expand_dims(img,                          axis=3)
    aggregated_featuremaps   = np.expand_dims(aggregated_featuremaps,       axis=3)
    img                      = np.concatenate((img,aggregated_featuremaps), axis=3)
    
    '''
    # Patch Extraction
    img, lungs_mask = extract_class_balanced_example_array(
            img, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )   
    '''

    # Patch Export
    trainX_hdf.create_dataset(subject_id, data=np.expand_dims(img[:,:,:,0], axis=3))     # 1 CT                         (Baseline Method)
    trainY_hdf.create_dataset(subject_id, data=img)                                      # 1 CT + AggregatedFeatureMap  (Standard Method)
    trainZ_hdf.create_dataset(subject_id, data=imgZ)                                     # 1 CT + 13FeatureMaps         (Experimental Method)

    print(label_name + 'CT_' + subject_id)

trainX_hdf.close()
trainY_hdf.close()
trainZ_hdf.close()









# Multiple Diseases

label_name     = 'MultipleDiseases/'
img_path       = '/DataFolder/lungs/Final_1593/train01/'                        + label_name
bm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/BM_1593/' + label_name
fm_path        = '/DataFolder/lungs/Final_1593/segmentation/densevnet/FM_1593/' + label_name
trainX_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployX/MultipleDiseases.h5', MODE)
trainY_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployY/MultipleDiseases.h5', MODE)
trainZ_hdf     =  h5py.File('/DataFolder/lungs/Final_1593/HDF5/deployZ/MultipleDiseases.h5', MODE)

# Create Directory List
files = []
for r, d, f in os.walk(img_path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Import Paths
    img_fn          = str(f)
    subject_id      = str(img_fn.split('.nii.gz')[0].split('CT_')[1])
    bm_fn           = str(bm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
    fm_fn           = str(fm_path + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')


    # Image
    img             = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
    img             = sitk.GetArrayFromImage(img)
    img             = whitening(img)                                 # Whitening Transformation (Variance=1)

    
    # Binary Masks
    mask            = sitk.GetArrayFromImage(sitk.ReadImage(bm_fn)).astype(np.int32)
    lungs_mask      = ((mask == 3)|(mask == 2)).astype(np.int32)
    body_mask       = ((mask != 0)).astype(np.int32)

    for i in range(0,lungs_mask.shape[0]):                          # Refine Lungs Mask (Remove Holes, Side-Components, Erode Lungs)
        lungs_mask[i,:,:] = ndimage.binary_dilation(lungs_mask[i,:,:]).astype(int)
        lungs_mask[i,:,:] = ndimage.binary_fill_holes(lungs_mask[i,:,:], structure=np.ones((2,2))).astype(int)
        label_im, nb_labels = ndimage.label(lungs_mask[i,:,:])
        sizes = ndimage.sum(lungs_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > L_MINAREA
        lungs_mask[i,:,:] = temp_mask[label_im]
        lungs_mask[i,:,:] = ndimage.binary_erosion(lungs_mask[i,:,:], structure=np.ones((10,10))).astype(int)

    for m in range(lungs_mask.shape[0]):                            # Remove Oversegmentation on Dead Slices
        if (img[m].mean() <= -0.90):
            lungs_mask[m] = 0                

    for i in range(0,body_mask.shape[0]):                           # Refine Full Body Mask (Remove Holes, Side-Components)
        body_mask[i,:,:] = ndimage.binary_dilation(body_mask[i,:,:]).astype(int)
        body_mask[i,:,:] = ndimage.binary_fill_holes(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)
        label_im, nb_labels = ndimage.label(body_mask[i,:,:])
        sizes = ndimage.sum(body_mask[i,:,:], label_im, range(nb_labels + 1))
        temp_mask = sizes > B_MINAREA
        body_mask[i,:,:] = temp_mask[label_im]
        body_mask[i,:,:] = ndimage.binary_erosion(body_mask[i,:,:], structure=np.ones((5,5))).astype(int)


    # Feature Maps
    file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(fm_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)                                                                  

    aoo = np.expand_dims(feature_maps[:,:,:,3]*body_mask,axis=3)    # Target 13 Feature Components
    boo = np.expand_dims(feature_maps[:,:,:,6]*body_mask,axis=3)
    coo = np.expand_dims(feature_maps[:,:,:,8]*body_mask,axis=3)
    doo = np.expand_dims(feature_maps[:,:,:,14]*body_mask,axis=3)
    eoo = np.expand_dims(feature_maps[:,:,:,16]*body_mask,axis=3)    
    foo = np.expand_dims(feature_maps[:,:,:,20]*body_mask,axis=3)
    goo = np.expand_dims(feature_maps[:,:,:,22]*body_mask,axis=3)
    hoo = np.expand_dims(feature_maps[:,:,:,25]*body_mask,axis=3)
    ioo = np.expand_dims(feature_maps[:,:,:,35]*body_mask,axis=3)
    joo = np.expand_dims(feature_maps[:,:,:,36]*body_mask,axis=3)
    koo = np.expand_dims(feature_maps[:,:,:,47]*body_mask,axis=3)
    loo = np.expand_dims(feature_maps[:,:,:,54]*body_mask,axis=3)
    moo = np.expand_dims(feature_maps[:,:,:,55]*body_mask,axis=3)

    # Aggregate and Normalize Feature Maps
    aggregated_featuremaps   = np.concatenate((aoo,boo,coo,doo,eoo,foo,goo,hoo,ioo,joo,koo,loo,moo),axis=3)
    aggregated_featuremaps   = np.mean(aggregated_featuremaps,axis=3)
    aggregated_featuremaps   = (aggregated_featuremaps-aggregated_featuremaps.min())/(aggregated_featuremaps.max()-aggregated_featuremaps.min())


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
    
    img                      = resize_image_with_crop_or_pad(img,                    [zdim,xdim,ydim], mode='symmetric')
    lungs_mask               = resize_image_with_crop_or_pad(lungs_mask,             [zdim,xdim,ydim], mode='symmetric')
    aggregated_featuremaps   = resize_image_with_crop_or_pad(aggregated_featuremaps, [zdim,xdim,ydim], mode='symmetric')



    # Alternate Aggregation (Experimental) --------------------------------------------------------------------------------------------------------
    aooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,3]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3)
    booZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,6]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    cooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,8]*body_mask,  [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    dooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,14]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    eooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,16]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    fooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,20]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    gooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,22]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    hooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,25]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    iooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,35]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    jooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,36]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
    kooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,47]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    looZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,54]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3) 
    mooZ  = np.expand_dims(resize_image_with_crop_or_pad(feature_maps[:,:,:,55]*body_mask, [zdim,xdim,ydim], mode='symmetric'), axis=3)
       
    imgZ                     = np.expand_dims(img,axis=3)
    aggregated_featuremapsZ  = np.concatenate((aooZ,booZ,cooZ,dooZ,eooZ,fooZ,gooZ,hooZ,iooZ,jooZ,kooZ,looZ,mooZ),axis=3)
    aggregated_featuremapsZ  = (aggregated_featuremapsZ-aggregated_featuremapsZ.min())/(aggregated_featuremapsZ.max()-aggregated_featuremapsZ.min())
    imgZ                     = np.concatenate((imgZ, aggregated_featuremapsZ),axis=3)

    '''
    imgZ, _ = extract_class_balanced_example_array(
            imgZ, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )
    '''
    # ----------------------------------------------------------------------------------------------------------------------------------------------



    # (3D -> 4D Conversion) -> Aggregated Input Volume
    img                      = np.expand_dims(img,                          axis=3)
    aggregated_featuremaps   = np.expand_dims(aggregated_featuremaps,       axis=3)
    img                      = np.concatenate((img,aggregated_featuremaps), axis=3)
    
    '''
    # Patch Extraction
    img, lungs_mask = extract_class_balanced_example_array(
            img, lungs_mask,
            example_size  = [PATCH, PATCH, PATCH],
            n_examples    =  PATCH_NUM,
            classes       =  1,
            class_weights = [1] )   
    '''

    # Patch Export
    trainX_hdf.create_dataset(subject_id, data=np.expand_dims(img[:,:,:,0], axis=3))     # 1 CT                         (Baseline Method)
    trainY_hdf.create_dataset(subject_id, data=img)                                      # 1 CT + AggregatedFeatureMap  (Standard Method)
    trainZ_hdf.create_dataset(subject_id, data=imgZ)                                     # 1 CT + 13FeatureMaps         (Experimental Method)

    print(label_name + 'CT_' + subject_id)


trainX_hdf.close()
trainY_hdf.close()
trainZ_hdf.close()
