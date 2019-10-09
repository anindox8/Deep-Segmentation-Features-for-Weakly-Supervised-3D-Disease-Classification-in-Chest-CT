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

    img_fn     = str(f)
    subject_id = '/DataFolder/segmentation/densevnet/probs/pneu/_' + str(img_fn.split('/')[-1].split('.')[0]) + '.nii.gz'
    subject_id = str(subject_id)

    img = sitk.ReadImage(img_fn)
    img = sitk.GetArrayFromImage(img)
    flip_img = np.flip(img,axis=2)    

    target_organ = sitk.GetImageFromArray(flip_img)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(subject_id)
    writer.Execute(target_organ)

    print('Saved: {}'.format(subject_id))
