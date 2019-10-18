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
import nibabel as nb

'''
 3D Binary Classification
 Preprocess: NIFTI-NIFTI Patch Extraction

 Update: 12/10/2019
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


disease_labels = ['MultipleDiseases/','Emphysema/','Mass/','Nodules/','Pneumonia-Atelectasis/']

for label_name in disease_labels:

    img_path       = '/DataFolder/lungs/final_dataset/temp/'                                 + label_name
    bm_path        = '/DataFolder/lungs/final_dataset/complete/segmentation/binary_masks/'   + label_name
    img1_save_path = '/DataFolder/lungs/final_dataset/complete/patches/patch1/original/'     + label_name
    img2_save_path = '/DataFolder/lungs/final_dataset/complete/patches/patch2/original/'     + label_name
    
    
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
    
    
        # Image
        imgITK          = sitk.ReadImage(img_fn, sitk.sitkFloat32)       
        imgNB           = nb.load(img_fn)
        img             = sitk.GetArrayFromImage(imgITK)
        img             = whitening(img)                                 # Whitening Transformation (Variance=1)
    
        
        # Binary Masks
        maskITK         = sitk.ReadImage(bm_fn)
        mask            = sitk.GetArrayFromImage(maskITK).astype(np.int32)
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
    
    
        # Patch Extraction
        img, lungs_mask = extract_class_balanced_example_array(
                np.expand_dims(img,axis=3), lungs_mask,
                example_size  = [PATCH, PATCH, PATCH],
                n_examples    =  PATCH_NUM,
                classes       =  1,
                class_weights = [1] )   
        img = np.squeeze(img,axis=4)
        np.swapaxes(img,1,3)
    
    
        # Save Patches
        imp1 = nb.Nifti1Image(img[0].T, imgNB.affine, imgNB.header)
        imp2 = nb.Nifti1Image(img[1].T, imgNB.affine, imgNB.header)
    
    
        nb.save(imp1, str(img1_save_path+'CT_'+subject_id+'.nii.gz'))
        nb.save(imp2, str(img2_save_path+'CT_'+subject_id+'.nii.gz'))
    
    
        print(label_name + 'CT_' + subject_id)

