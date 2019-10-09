import os
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import KFold



'''
 Binary Classification (3D)
 Feed: Generating Metadata

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



# Normal Lungs
disease_name       = 'NormalLungPatient'
folds              = 4
patch_folder       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Normal/'
mask_folder        = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Normal/_'
fm_folder          = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Normal.h5'
label_sign         = 0

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
mask_path_list     = []
fm_path_list       = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    # Extracting Names
    ct_volume_name       = patient_list[i]
    subject_id           = ct_volume_name.split('.nii.gz')[0]
    name_split           = ct_volume_name.split('.nii.gz')[0].split('CT_')
    mask_volume_name     = name_split[1] +'_niftynet_out.nii.gz'

    # Setting Path
    patch_path           = patch_folder + patient_list[i]
    mask_path            = mask_folder  + mask_volume_name
    fm_path              = fm_folder

    # Updating Lists
    patient_path_list.append(patch_path)
    mask_path_list.append(mask_path)
    fm_path_list.append(fm_path)
    subject_id_list.append(subject_id)

# Synchronous Data Shuffle
a, b, c, d, e  = shuffle(patient_path_list, mask_path_list, fm_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c,d,e)),
columns           = ['img','binary_mask','feature_map','lbl','subject_id'])
img               = shuffled_dataset['img']
binary_mask       = shuffled_dataset['binary_mask']
feature_map       = shuffled_dataset['feature_map']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = binary_mask[train]
    c_train = feature_map[train]
    d_train = lbl[train]
    e_train = subject_id[train]

    a_val = img[val]
    b_val = binary_mask[val]
    c_val = feature_map[val]
    d_val = lbl[val]
    e_val = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train,d_train,e_train)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    trainData_name             = 'csvIII/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val,d_val,e_val)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    valData_name               = 'csvIII/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)











# Diseased Lungs (Edema)
disease_name       = 'EdemaPatient'
folds              = 4
patch_folder       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Edema/'
mask_folder        = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/edema/_'
fm_folder          = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Edema.h5'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
mask_path_list     = []
fm_path_list       = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    # Extracting Names
    ct_volume_name       = patient_list[i]
    subject_id           = ct_volume_name.split('.nii.gz')[0]
    name_split           = ct_volume_name.split('.nii.gz')[0].split('CT_')
    mask_volume_name     = name_split[1] +'_niftynet_out.nii.gz'

    # Setting Path
    patch_path           = patch_folder + patient_list[i]
    mask_path            = mask_folder  + mask_volume_name
    fm_path              = fm_folder

    # Updating Lists
    patient_path_list.append(patch_path)
    mask_path_list.append(mask_path)
    fm_path_list.append(fm_path)
    subject_id_list.append(subject_id)

# Synchronous Data Shuffle
a, b, c, d, e  = shuffle(patient_path_list, mask_path_list, fm_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c,d,e)),
columns           = ['img','binary_mask','feature_map','lbl','subject_id'])
img               = shuffled_dataset['img']
binary_mask       = shuffled_dataset['binary_mask']
feature_map       = shuffled_dataset['feature_map']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = binary_mask[train]
    c_train = feature_map[train]
    d_train = lbl[train]
    e_train = subject_id[train]

    a_val = img[val]
    b_val = binary_mask[val]
    c_val = feature_map[val]
    d_val = lbl[val]
    e_val = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train,d_train,e_train)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    trainData_name             = 'csvIII/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val,d_val,e_val)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    valData_name               = 'csvIII/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)











# Diseased Lungs (Atelectasis)
disease_name       = 'AtelectasisPatient'
folds              = 4
patch_folder       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Atlectasis/'
mask_folder        = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Atelectasis/_'
fm_folder          = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Atelectasis.h5'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
mask_path_list     = []
fm_path_list       = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    # Extracting Names
    ct_volume_name       = patient_list[i]
    subject_id           = ct_volume_name.split('.nii.gz')[0]
    name_split           = ct_volume_name.split('.nii.gz')[0].split('CT_')
    mask_volume_name     = name_split[1] +'_niftynet_out.nii.gz'

    # Setting Path
    patch_path           = patch_folder + patient_list[i]
    mask_path            = mask_folder  + mask_volume_name
    fm_path              = fm_folder

    # Updating Lists
    patient_path_list.append(patch_path)
    mask_path_list.append(mask_path)
    fm_path_list.append(fm_path)
    subject_id_list.append(subject_id)

# Synchronous Data Shuffle
a, b, c, d, e  = shuffle(patient_path_list, mask_path_list, fm_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c,d,e)),
columns           = ['img','binary_mask','feature_map','lbl','subject_id'])
img               = shuffled_dataset['img']
binary_mask       = shuffled_dataset['binary_mask']
feature_map       = shuffled_dataset['feature_map']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = binary_mask[train]
    c_train = feature_map[train]
    d_train = lbl[train]
    e_train = subject_id[train]

    a_val = img[val]
    b_val = binary_mask[val]
    c_val = feature_map[val]
    d_val = lbl[val]
    e_val = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train,d_train,e_train)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    trainData_name             = 'csvIII/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val,d_val,e_val)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    valData_name               = 'csvIII/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)











# Diseased Lungs (Pneumonia)
disease_name       = 'PneumoniaPatient'
folds              = 4
patch_folder       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Pneumonia/'
mask_folder        = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Pneumonia/_'
fm_folder          = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Pneumonia.h5'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
mask_path_list     = []
fm_path_list       = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    # Extracting Names
    ct_volume_name       = patient_list[i]
    subject_id           = ct_volume_name.split('.nii.gz')[0]
    name_split           = ct_volume_name.split('.nii.gz')[0].split('CT_')
    mask_volume_name     = name_split[1] +'_niftynet_out.nii.gz'

    # Setting Path
    patch_path           = patch_folder + patient_list[i]
    mask_path            = mask_folder  + mask_volume_name
    fm_path              = fm_folder 

    # Updating Lists
    patient_path_list.append(patch_path)
    mask_path_list.append(mask_path)
    fm_path_list.append(fm_path)
    subject_id_list.append(subject_id)

# Synchronous Data Shuffle
a, b, c, d, e  = shuffle(patient_path_list, mask_path_list, fm_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c,d,e)),
columns           = ['img','binary_mask','feature_map','lbl','subject_id'])
img               = shuffled_dataset['img']
binary_mask       = shuffled_dataset['binary_mask']
feature_map       = shuffled_dataset['feature_map']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = binary_mask[train]
    c_train = feature_map[train]
    d_train = lbl[train]
    e_train = subject_id[train]

    a_val = img[val]
    b_val = binary_mask[val]
    c_val = feature_map[val]
    d_val = lbl[val]
    e_val = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train,d_train,e_train)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    trainData_name             = 'csvIII/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val,d_val,e_val)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    valData_name               = 'csvIII/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)














# Diseased Lungs (Nodules)
disease_name       = 'NodulesPatient'
folds              = 4
patch_folder       = '/DataFolder/lungs/original_volumes/all_Segmentation_data/Nodules/'
mask_folder        = '/DataFolder/lungs/segmentation/densevnet/binary_masks/diseases_Training_AllPrediction/Nodules/_'
fm_folder          = '/DataFolder/lungs/segmentation/densevnet/feature_maps/Aggregated_Feature-Maps_Spatial-Priori/Nodules.h5'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
mask_path_list     = []
fm_path_list       = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    # Extracting Names
    ct_volume_name       = patient_list[i]
    subject_id           = ct_volume_name.split('.nii.gz')[0]
    name_split           = ct_volume_name.split('.nii.gz')[0].split('CT_')
    mask_volume_name     = name_split[1] +'_niftynet_out.nii.gz'

    # Setting Path
    patch_path           = patch_folder + patient_list[i]
    mask_path            = mask_folder  + mask_volume_name
    fm_path              = fm_folder  

    # Updating Lists
    patient_path_list.append(patch_path)
    mask_path_list.append(mask_path)
    fm_path_list.append(fm_path)
    subject_id_list.append(subject_id)

# Synchronous Data Shuffle
a, b, c, d, e  = shuffle(patient_path_list, mask_path_list, fm_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c,d,e)),
columns           = ['img','binary_mask','feature_map','lbl','subject_id'])
img               = shuffled_dataset['img']
binary_mask       = shuffled_dataset['binary_mask']
feature_map       = shuffled_dataset['feature_map']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = binary_mask[train]
    c_train = feature_map[train]
    d_train = lbl[train]
    e_train = subject_id[train]

    a_val = img[val]
    b_val = binary_mask[val]
    c_val = feature_map[val]
    d_val = lbl[val]
    e_val = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train,d_train,e_train)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    trainData_name             = 'csvIII/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val,d_val,e_val)),
    columns                    = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    valData_name               = 'csvIII/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)
