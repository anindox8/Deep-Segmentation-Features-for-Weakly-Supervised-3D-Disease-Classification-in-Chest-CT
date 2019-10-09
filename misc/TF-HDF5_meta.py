from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import numpy as np
import h5py


'''
 3D Binary Classification
 Merging All Data to HDF5

 Update: 31/07/2019
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
print('Loading Normal')
features_hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Normal.h5', 'r')
out_features_hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/TF_Features.h5', 'w')
keys = list(features_hf.keys())

all_features = np.zeros(shape=(1,112,112,112,2))
for i in range(0,len(keys)):
   features = np.array(features_hf.get(keys[i]))
   all_features = np.concatenate((all_features,features),axis=0)
all_features = all_features[1:,:,:,:,:]
features_hf.close()

out_features_hf.create_dataset('all_normal', data=all_features)
print('Complete')


# Edema
print('Loading Edema')
features_hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Edema.h5', 'r')
keys = list(features_hf.keys())

all_features = np.zeros(shape=(1,112,112,112,2))
for i in range(0,len(keys)):
   features = np.array(features_hf.get(keys[i]))
   all_features = np.concatenate((all_features,features),axis=0)
all_features = all_features[1:,:,:,:,:]
features_hf.close()

out_features_hf.create_dataset('all_edema', data=all_features)
print('Complete')



# Pneumonia
print('Loading Pneumonia')
features_hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Pneumonia.h5', 'r')
keys = list(features_hf.keys())

all_features = np.zeros(shape=(1,112,112,112,2))
for i in range(0,len(keys)):
   features = np.array(features_hf.get(keys[i]))
   all_features = np.concatenate((all_features,features),axis=0)
all_features = all_features[1:,:,:,:,:]
features_hf.close()

out_features_hf.create_dataset('all_pneumonia', data=all_features)
print('Complete')




# Nodules
print('Loading Nodules')
features_hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Nodules.h5', 'r')
keys = list(features_hf.keys())

all_features = np.zeros(shape=(1,112,112,112,2))
for i in range(0,len(keys)):
   features = np.array(features_hf.get(keys[i]))
   all_features = np.concatenate((all_features,features),axis=0)
all_features = all_features[1:,:,:,:,:]
features_hf.close()

out_features_hf.create_dataset('all_nodules', data=all_features)
print('Complete')





# Atelectasis
print('Loading Atelectasis')
features_hf = h5py.File('/DataFolder/lungs/segmentation/densevnet/feature_maps/HDF5/Atelectasis.h5', 'r')
keys = list(features_hf.keys())

all_features = np.zeros(shape=(1,112,112,112,2))
for i in range(0,len(keys)):
   features = np.array(features_hf.get(keys[i]))
   all_features = np.concatenate((all_features,features),axis=0)
all_features = all_features[1:,:,:,:,:]
features_hf.close()

out_features_hf.create_dataset('all_atelectasis', data=all_features)

out_features_hf.close()
print('Complete')