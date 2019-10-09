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

'''
 3D Binary Classification
 Preprocess: Aggregate Multi-Res Segmentation Feature Maps

 Update: 30/07/2019
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


# Normal
path = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Normal/'

# Create Directory List
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/MIP_Feature-Maps_Spatial-Priori/Normal.h5', 'w')

for f in files:

    # Load Feature_Maps Image
    img_fn     = str(f)
    subject_id = 'CT' + str(img_fn.split('seg')[-1].split('__niftynet')[0])
    subject_id = str(subject_id)


    # Load Organ Probabilities Map (Spatial Priori)
    img0_fn     = str('/DataFolder/lungs/segmentation/densevnet/probs_masks/Normal/__' + str(subject_id.split('CT_')[-1]) + '_niftynet_out.nii.gz')
    probs_mask  = sitk.ReadImage(img0_fn, sitk.sitkFloat32)
    probs_mask  = sitk.GetArrayFromImage(probs_mask)
    probs_mask  = np.flip(probs_mask,axis=1)
    probs_mask  = np.expand_dims(probs_mask,axis=3)


    # Load Segmentation Feature Maps
    file_reader = sitk.ImageFileReader()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)
    

    print('Probabilities Mask Shape: {}'.format(probs_mask.shape))
    print('Feature Maps Shape: {}'.format(feature_maps.shape))


    # Aggregate Feature Maps + Spatial Priori
    feature_maps = feature_maps/feature_maps.max()
    bottle       = np.concatenate((feature_maps,probs_mask), axis=3)
    bottle       = np.mean(bottle, axis=3)

    print('Aggregated Feature Map Shape: {}'.format(bottle.shape))

    # Save Feature Maps
    hf.create_dataset(subject_id,data=bottle)

    print('Saved: {}'.format(subject_id))
hf.close()










# Edema
path = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Edema/'

# Create Directory List
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/MIP_Feature-Maps_Spatial-Priori/Edema.h5', 'w')

for f in files:

    # Load Feature_Maps Image
    img_fn     = str(f)
    subject_id = 'CT' + str(img_fn.split('seg')[-1].split('__niftynet')[0])
    subject_id = str(subject_id)


    # Load Organ Probabilities Map (Spatial Priori)
    img0_fn     = str('/DataFolder/lungs/segmentation/densevnet/probs_masks/Edema/__' + str(subject_id.split('CT_')[-1]) + '_niftynet_out.nii.gz')
    probs_mask  = sitk.ReadImage(img0_fn, sitk.sitkFloat32)
    probs_mask  = sitk.GetArrayFromImage(probs_mask)
    probs_mask  = np.flip(probs_mask,axis=1)
    probs_mask  = np.expand_dims(probs_mask,axis=3)


    # Load Segmentation Feature Maps
    file_reader = sitk.ImageFileReader()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)
    

    print('Probabilities Mask Shape: {}'.format(probs_mask.shape))
    print('Feature Maps Shape: {}'.format(feature_maps.shape))


    # Aggregate Feature Maps + Spatial Priori
    feature_maps = feature_maps/feature_maps.max()
    bottle       = np.concatenate((feature_maps,probs_mask), axis=3)
    bottle       = np.mean(bottle, axis=3)

    print('Aggregated Feature Map Shape: {}'.format(bottle.shape))

    # Save Feature Maps
    hf.create_dataset(subject_id,data=bottle)

    print('Saved: {}'.format(subject_id))
hf.close()












# Pneumonia
path = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Pneumonia/'

# Create Directory List
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/MIP_Feature-Maps_Spatial-Priori/Pneumonia.h5', 'w')

for f in files:

    # Load Feature_Maps Image
    img_fn     = str(f)
    subject_id = 'CT' + str(img_fn.split('seg')[-1].split('__niftynet')[0])
    subject_id = str(subject_id)


    # Load Organ Probabilities Map (Spatial Priori)
    img0_fn     = str('/DataFolder/lungs/segmentation/densevnet/probs_masks/Pneumonia/__' + str(subject_id.split('CT_')[-1]) + '_niftynet_out.nii.gz')
    probs_mask  = sitk.ReadImage(img0_fn, sitk.sitkFloat32)
    probs_mask  = sitk.GetArrayFromImage(probs_mask)
    probs_mask  = np.flip(probs_mask,axis=1)
    probs_mask  = np.expand_dims(probs_mask,axis=3)


    # Load Segmentation Feature Maps
    file_reader = sitk.ImageFileReader()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)
    

    print('Probabilities Mask Shape: {}'.format(probs_mask.shape))
    print('Feature Maps Shape: {}'.format(feature_maps.shape))


    # Aggregate Feature Maps + Spatial Priori
    feature_maps = feature_maps/feature_maps.max()
    bottle       = np.concatenate((feature_maps,probs_mask), axis=3)
    bottle       = np.mean(bottle, axis=3)

    print('Aggregated Feature Map Shape: {}'.format(bottle.shape))

    # Save Feature Maps
    hf.create_dataset(subject_id,data=bottle)

    print('Saved: {}'.format(subject_id))
hf.close()














# Nodules
path = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Nodules/'

# Create Directory List
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/MIP_Feature-Maps_Spatial-Priori/Nodules.h5', 'w')

for f in files:

    # Load Feature_Maps Image
    img_fn     = str(f)
    subject_id = 'CT' + str(img_fn.split('seg')[-1].split('__niftynet')[0])
    subject_id = str(subject_id)


    # Load Organ Probabilities Map (Spatial Priori)
    img0_fn     = str('/DataFolder/lungs/segmentation/densevnet/probs_masks/Nodules/__' + str(subject_id.split('CT_')[-1]) + '_niftynet_out.nii.gz')
    probs_mask  = sitk.ReadImage(img0_fn, sitk.sitkFloat32)
    probs_mask  = sitk.GetArrayFromImage(probs_mask)
    probs_mask  = np.flip(probs_mask,axis=1)
    probs_mask  = np.expand_dims(probs_mask,axis=3)


    # Load Segmentation Feature Maps
    file_reader = sitk.ImageFileReader()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)
    

    print('Probabilities Mask Shape: {}'.format(probs_mask.shape))
    print('Feature Maps Shape: {}'.format(feature_maps.shape))


    # Aggregate Feature Maps + Spatial Priori
    feature_maps = feature_maps/feature_maps.max()
    bottle       = np.concatenate((feature_maps,probs_mask), axis=3)
    bottle       = np.mean(bottle, axis=3)

    print('Aggregated Feature Map Shape: {}'.format(bottle.shape))

    # Save Feature Maps
    hf.create_dataset(subject_id,data=bottle)

    print('Saved: {}'.format(subject_id))
hf.close()













# Atelectasis
path = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Atelectasis/'

# Create Directory List
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.nii.gz' in file:
            files.append(os.path.join(r, file))

hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/MIP_Feature-Maps_Spatial-Priori/Atelectasis.h5', 'w')

for f in files:

    # Load Feature_Maps Image
    img_fn     = str(f)
    subject_id = 'CT' + str(img_fn.split('seg')[-1].split('__niftynet')[0])
    subject_id = str(subject_id)


    # Load Organ Probabilities Map (Spatial Priori)
    img0_fn     = str('/DataFolder/lungs/segmentation/densevnet/probs_masks/Atelectasis/__' + str(subject_id.split('CT_')[-1]) + '_niftynet_out.nii.gz')
    probs_mask  = sitk.ReadImage(img0_fn, sitk.sitkFloat32)
    probs_mask  = sitk.GetArrayFromImage(probs_mask)
    probs_mask  = np.flip(probs_mask,axis=1)
    probs_mask  = np.expand_dims(probs_mask,axis=3)


    # Load Segmentation Feature Maps
    file_reader = sitk.ImageFileReader()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_fn)
    file_reader.ReadImageInformation()
    feature_maps_size = list(file_reader.GetSize())
    file_reader.SetExtractSize([0 if v == 1 else v for v in feature_maps_size])
    feature_maps = file_reader.Execute()
    feature_maps = sitk.Compose( [sitk.Extract(feature_maps, feature_maps.GetSize()[:3]+(0,), [0,0,0, i]) for i in range(feature_maps_size[-1])] )
    feature_maps = sitk.GetArrayFromImage(feature_maps)
    

    print('Probabilities Mask Shape: {}'.format(probs_mask.shape))
    print('Feature Maps Shape: {}'.format(feature_maps.shape))


    # Aggregate Feature Maps + Spatial Priori
    feature_maps = feature_maps/feature_maps.max()
    bottle       = np.concatenate((feature_maps,probs_mask), axis=3)
    bottle       = np.mean(bottle, axis=3)

    print('Aggregated Feature Map Shape: {}'.format(bottle.shape))

    # Save Feature Maps
    hf.create_dataset(subject_id,data=bottle)

    print('Saved: {}'.format(subject_id))
hf.close()