
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
from dltk.io.preprocessing import whitening
import pandas as pd
import numpy as np
from PIL import Image
import scipy.misc as sci



'''
 3D Binary Classification
 Visuals: Lungs Segmentation Masks

 Update: 17/07/2019
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


# Training Images
train_filenames = pd.read_csv(
    '/Local/scripts/lungs/classification/feed/csv/Lung_CV-Training-Fold-1.csv', dtype=object, keep_default_na=False,
    na_values=[]).values

counter = 1

for f in train_filenames:

    # Column 1: Image
    img_fn     = str(f[0])
    subject_id = img_fn.split('/')[-1].split('.')[0]

    # Loading NIFTI Images via SimpleITK
    img_sitk = sitk.ReadImage(img_fn, sitk.sitkFloat32)
    images   = sitk.GetArrayFromImage(img_sitk)
    
    
    # Loading Labels/GT
    mask_fn = str(f[2])
    mask    = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn)).astype(np.int32)
    mask = ((mask == 3)|(mask == 2)).astype(np.int32)

    # Locate Maximum Area Mask
    peak = 0
    for m in range(mask.shape[0]):
        maxint = mask[m].sum()
        if (maxint>=peak):
            peak       = maxint
            peakslice  = m

    # Mask Input Image
    masked_image = mask*images
    print(str(subject_id) +' Mask: '+str(masked_image.shape) +' Peak Slice: ' +str(peakslice) +' Max Area (Pixels): ' +str(peak))
    masked_lungs = masked_image[peakslice]

    # Save Figures
    sci.toimage(masked_lungs).save('./lungs/{}.png'.format(subject_id))
    
    counter = counter + 1 




# Validation Images
train_filenames = pd.read_csv(
    '/Local/scripts/lungs/classification/feed/csv/Lung_CV-Validation-Fold-1.csv', dtype=object, keep_default_na=False,
    na_values=[]).values


for f in train_filenames:

    # Column 1: Image
    img_fn     = str(f[0])
    subject_id = img_fn.split('/')[-1].split('.')[0]

    # Loading NIFTI Images via SimpleITK
    img_sitk = sitk.ReadImage(img_fn, sitk.sitkFloat32)
    images   = sitk.GetArrayFromImage(img_sitk)
    
    
    # Loading Labels/GT
    mask_fn = str(f[2])
    mask    = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn)).astype(np.int32)
    mask = ((mask == 3)|(mask == 2)).astype(np.int32)

    # Locate Maximum Area Mask
    peak = 0
    for m in range(mask.shape[0]):
        maxint = mask[m].sum()
        if (maxint>=peak):
            peak       = maxint
            peakslice  = m

    # Mask Input Image
    masked_image = mask*images
    print(str(subject_id) +' Mask: '+str(masked_image.shape) +' Peak Slice: ' +str(peakslice) +' Max Area (Pixels): ' +str(peak))
    masked_lungs = masked_image[peakslice]

    # Save Figures
    sci.toimage(masked_lungs).save('./lungs/{}.png'.format(subject_id))
    
    counter = counter + 1 