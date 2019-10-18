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
 Preprocess: NPY Consolidator

 Update: 15/10/2019
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


disease_labels = ['MultipleDiseases/','Emphysema/','Mass/','Nodules/','Pneumonia-Atelectasis/','Normal/']

for label_name in disease_labels:

    patch1_img_path    = '/DataFolder/lungs/final_dataset/complete/patches/patch1/original/'     + label_name
    patch2_img_path    = '/DataFolder/lungs/final_dataset/complete/patches/patch2/original/'     + label_name
    patch1_fm_path     = '/DataFolder/lungs/final_dataset/complete/patches/patch1/feature_maps/' + label_name
    patch2_fm_path     = '/DataFolder/lungs/final_dataset/complete/patches/patch2/feature_maps/' + label_name
    save_path          = '/DataFolder/lungs/final_dataset/complete/patches/numpy/'               + label_name
    
    
    # Create Directory List
    files = []
    for r, d, f in os.walk(patch1_img_path):
        for file in f:
            if '.nii.gz' in file:
                files.append(os.path.join(r, file))
    
    for f in files:
        # Assign Paths
        img1_fn          = str(f)
        subject_id       = str(img1_fn.split('.nii.gz')[0].split('CT_')[1])
        img2_fn          = str(patch1_img_path + 'CT_'         + subject_id + '.nii.gz')
        fm1_fn           = str(patch1_fm_path  + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')
        fm2_fn           = str(patch2_fm_path  + 'window_seg_' + subject_id + '__niftynet_out.nii.gz')

        op_path          = str(save_path+'CT_'+subject_id+'.npy')
        if not os.path.exists(op_path):
    
            # Load CT Patches
            img1ITK          = sitk.ReadImage(img1_fn, sitk.sitkFloat32)
            img2ITK          = sitk.ReadImage(img2_fn, sitk.sitkFloat32)
            img1             = sitk.GetArrayFromImage(img1ITK)
            img2             = sitk.GetArrayFromImage(img2ITK)
        
        
            # Feature Maps
            file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(fm1_fn)
            file_reader.ReadImageInformation()
            feature_maps_size = list(file_reader.GetSize())
            file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
            fm1         = file_reader.Execute()
            fm1         = sitk.Compose( [sitk.Extract(fm1, fm1.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
            fm1         = sitk.GetArrayFromImage(fm1)                    
        
        
            file_reader = sitk.ImageFileReader()                             # 5D -> 4D Conversion
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(fm2_fn)
            file_reader.ReadImageInformation()
            feature_maps_size = list(file_reader.GetSize())
            file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
            fm2         = file_reader.Execute()
            fm2         = sitk.Compose( [sitk.Extract(fm2, fm2.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
            fm2         = sitk.GetArrayFromImage(fm2)                      
        
        
            # Reshape and Concatenate
            img1   =  np.expand_dims(img1,axis=3)
            img2   =  np.expand_dims(img2,axis=3)
            op1    =  np.concatenate((img1, fm1[:,:,:,1:38]),axis=3)         # 1 CT + 60 Feature Maps
            op2    =  np.concatenate((img2, fm2[:,:,:,1:38]),axis=3)         # 1 CT + 60 Feature Maps
            op1    =  np.expand_dims(op1,axis=0)
            op2    =  np.expand_dims(op2,axis=0)
            op     =  np.concatenate((op1,op2),axis=0)
        
            # Export NPY
            np.save(op_path, op)
            print(str(label_name+'CT_'+subject_id+'.npy'))
        else:
            print(str('Skipping:'+label_name+'CT_'+subject_id+'.npy'))
