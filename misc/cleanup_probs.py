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

'''
 3D Binary Classification
 Visuals: Lungs Segmentation Probability Masks Cleanup

 Update: 26/07/2019
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

path = '/DataFolder/segmentation/densevnet/probs/Pneumonia/'

# Create Directory List
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))


for f in files:

    # Load Image
    img_fn     = str(f)
    subject_id = '/DataFolder/segmentation/densevnet/probs/pneu/_' + str(img_fn.split('/')[-1].split('.')[0]) + '.nii.gz'
    subject_id = str(subject_id)

    # Loading 5D NIFTI Images as 4D Image
    file_reader = sitk.ImageFileReader()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_fn)
    file_reader.ReadImageInformation()
    img_size    = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in img_size])
    img = file_reader.Execute()
    segmentation_map =  sitk.Compose( [sitk.Extract(img, img.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(img_size[-1])] )
    segmentation_map = sitk.GetArrayFromImage(segmentation_map)
    
    # Extract Target Organ Channels
    target_organ = segmentation_map[:,:,:,2] + segmentation_map[:,:,:,3]
    
    # Loading Labels/GT
    target_organ = sitk.GetImageFromArray(target_organ)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(subject_id)
    writer.Execute(target_organ)

    print('Saved: {}'.format(subject_id))
