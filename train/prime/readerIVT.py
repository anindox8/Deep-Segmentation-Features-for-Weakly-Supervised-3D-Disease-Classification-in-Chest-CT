from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import os
import numpy as np
from dltk.io.augmentation import flip
import h5py
import time
 

'''
 3D Binary Classification

 Update: 18/08/2019
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

        # Load Data
        data_hf  = h5py.File(f[0], 'r')
        img      = np.array(data_hf.get(f[2]))
        data_hf.close()                                         

        # Load Labels
        lbl  = np.int(f[1])
        lbl  = np.expand_dims(lbl, axis=-1).astype(np.int32)
    
        # Training Mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            pass

        # Augmentation Flag    
        if params['augmentation']:
            img = _augment(img)    

        print('Loaded Label = {}; Time = {}'.format(f[1],(time.time()-t0)))

        # Return Training Patches
        for e in range(params['n_patches']):                     # Permitted Max 2 Patches for Std Optimized Datasets
            yield {'features': {'x': img[e].astype(np.float32)},
                   'labels':   {'y': lbl.astype(np.float32)},
                   'img_id':         f[1]}
    return
