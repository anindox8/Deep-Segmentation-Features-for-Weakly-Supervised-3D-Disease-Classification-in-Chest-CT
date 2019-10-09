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
from patch import extract_class_balanced_example_array
import h5py
import time
 

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

** Temmporary Reader File Optimized for HDF5 Files
   and Conventional csvIIIh Feeders

   # PATCH SIZE:        [128, 128, 128]
   # NUMBER OF PATCHES:  2
   # FEATURE MAPS:       AVG + SPATIAL PRIORI 

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

        t0 = time.time()  

        # Load Features
        features_hf  = h5py.File(f[0], 'r')
        img = np.array(features_hf.get(f[2]))                                         

        # Load Labels
        lbl  = np.int(f[1])
        lbl  = np.expand_dims(lbl, axis=-1).astype(np.int32)
    
        # Training Mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            pass

        # Augmentation Flag    
        if params['augmentation']:
            img = _augment(img)    

        print('Loaded {}; Time = {}'.format(f[2],(time.time()-t0)))

        # Return Training Patches
        for e in range(params['n_patches']):                     # Permitted max 2 for Std Data
            yield {'features': {'x': img[e].astype(np.float32)},
                   'labels':   {'y': lbl.astype(np.float32)},
                   'img_id':         f[2]}
    return
