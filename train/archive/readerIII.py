from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
from dltk.io.augmentation import flip
from dltk.io.preprocessing import whitening
from dltk.io.preprocessing import resize_image_with_crop_or_pad
# from dltk.io.augmentation import extract_class_balanced_example_array
from patch import extract_class_balanced_example_array
import h5py
 

'''
 3D Binary Classification
 Train: Preprocessing Volumes (Import, Pad, Whiten, Concatenate)

 Update: 29/07/2019
 Contributors: ft42, as1044
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Edema
     - Atelectasis
     - Pneumonia
     - Nodules
'''


def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.
    Args:
        file_references (list):
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
            PREDICT.
        params (dict, optional): A dictionary to parametrise read_fn outputs
            (e.g. reader_params = {'n_patches': 10, 'patch_size':
            [64, 64, 64], 'extract_patches': True}, etc.).
    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """

    def _augment(img):
        return flip(img, axis=2)

    for f in file_references:
        
        # Column 1: Image
        img_fn     = str(f[0])
        subject_id = img_fn.split('/')[-1].split('.')[0]
        img        = sitk.ReadImage(img_fn, sitk.sitkFloat32)       # Loading NIFTI Images via SimpleITK
        img        = sitk.GetArrayFromImage(img)
        img        = whitening(img)                                 # Whitening Transformation (Variance=1)
        
        # Column 2: Binary Mask
        binary_mask_fn  = str(f[1])
        binary_mask     = sitk.GetArrayFromImage(sitk.ReadImage(binary_mask_fn)).astype(np.int32)
        
        
        # Column 3: Aggregated Feature Maps
        hf = h5py.File(f[2], 'r')
        feature_maps = np.array(hf.get(f[4]))                                                                       
        hf.close()

        # Padding Images
        patch_size = params['patch_size']
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
        feature_maps    = resize_image_with_crop_or_pad(feature_maps, [zdim,xdim,ydim], mode='symmetric')
        
        


        # Column 5: Label
        lbl         = np.int(f[3])
        lbl         = np.expand_dims(lbl, axis=-1).astype(np.int32)
        
        # Expand to 4D
        img          = np.expand_dims(img, axis=3)
        feature_maps = np.expand_dims(feature_maps, axis=3)


        # Aggregated Input
        img = np.concatenate((img,feature_maps),axis=3)


        # TensorFlow Mode-Based Execution
        # Prediction Mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': img},
                   'labels':   {'y': lbl.astype(np.float32)}, 
                   'img_id':         subject_id}
        
        # Training Mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            pass
        
        # Augmentation Flag    
        if params['augmentation']:
            img = _augment(img)

        # Return Training Examples
        if params['extract_patches']:
            img,binary_mask = extract_class_balanced_example_array(
                img,binary_mask,
                example_size  = params['patch_size'],
                n_examples    = params['n_patches'],
                classes       = 4, class_weights=[0,0,1,1]   # Label 3,4 => Right,Left Lungs (XCAT)
                )

            for e in range(params['n_patches']):
                yield {'features': {'x': img[e].astype(np.float32)},
                       'labels':   {'y': lbl.astype(np.float32)},
                       'img_id':         subject_id}

        # Return Full Images
        else:
            yield {'features': {'x': img},
                   'labels':   {'y': lbl.astype(np.float32)},
                   'img_id':         subject_id}

    return
