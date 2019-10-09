from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import os
import numpy as np
from dltk.io.augmentation import flip
from patch import extract_class_balanced_example_array
import h5py
import time
from dltk.io.preprocessing import resize_image_with_crop_or_pad
from scipy import ndimage
from sklearn.preprocessing import normalize
from skimage.measure import label
import SimpleITK as sitk

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
        if params['extract_patches']:
            data_hf  = h5py.File(f[0], 'r')
            img      = np.array(data_hf.get(f[2]))
            data_hf.close()    
        else:    
            if   params['model_type']=='X':
                data_hf  = h5py.File(str(str(f[0]).split('trainX/')[0]+'deployX/'+str(f[0]).split('trainX/')[1]), 'r')    
                img      = np.array(data_hf.get(f[2]))
            elif params['model_type']=='Y':
                data_hf  = h5py.File(str(str(f[0]).split('trainY/')[0]+'deployY/'+str(f[0]).split('trainY/')[1]), 'r')    
                img      = np.array(data_hf.get(f[2]))
            elif params['model_type']=='Z':
                data_hf  = h5py.File(str(str(f[0]).split('trainZ/')[0]+'deployZ/'+str(f[0]).split('trainZ/')[1]), 'r')    
                img      = np.array(data_hf.get(f[2]))
            data_hf.close()


        # Load Labels
        lbl  = np.int(f[1])
        lbl  = np.expand_dims(lbl, axis=-1).astype(np.int32)

    
        # TensorFlow Mode-Based Execution
        # Prediction Mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': img.astype(np.float32)},
                   'labels':   {'y': lbl.astype(np.float32)},
                   'img_id':         str(f[2])}
    return

