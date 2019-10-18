import os
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import KFold



'''
 Binary Classification (3D)
 Feed: Generating Metadata

 Update: 16/10/2019
 Contributors: ft42, as1044
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Emphysema
     - Pneumonia-Atelectasis
     - Mass
     - Nodules
'''


# Normal (Lungs)
disease_name       = 'NormalLungsPatient'
folds              = 5
patch_folder       = '/DataFolder/lungs/final_dataset/complete/NIfTI/train-val/Normal/'
label_sign         = 0

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    
    # Extracting Names
    subject_id           = patient_list[i].split('.nii.gz')[0].split('CT_')[1]
    subject_id_list.append(subject_id)
    # Setting I/O Path
    patient_path_list.append(patch_folder + patient_list[i])

# Synchronous Data Shuffle
a, b, c  = shuffle(patient_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
columns           = ['img', 'lbl','subject_id'])
img               = shuffled_dataset['img']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = lbl[train]
    c_train = subject_id[train]
    a_val   = img[val]
    b_val   = lbl[val]
    c_val   = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['img', 'lbl', 'subject_id'])
    trainData_name             = 'csv_v2/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['img', 'lbl', 'subject_id'])
    valData_name               = 'csv_v2/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)













# Emphysema (Lungs)
disease_name       = 'EmphysemaPatient'
folds              = 5
patch_folder       = '/DataFolder/lungs/final_dataset/complete/NIfTI/train-val/Emphysema/'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    
    # Extracting Names
    subject_id           = patient_list[i].split('.nii.gz')[0].split('CT_')[1]
    subject_id_list.append(subject_id)
    # Setting I/O Path
    patient_path_list.append(patch_folder + patient_list[i])

# Synchronous Data Shuffle
a, b, c  = shuffle(patient_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
columns           = ['img', 'lbl','subject_id'])
img               = shuffled_dataset['img']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = lbl[train]
    c_train = subject_id[train]
    a_val   = img[val]
    b_val   = lbl[val]
    c_val   = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['img', 'lbl', 'subject_id'])
    trainData_name             = 'csv_v2/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['img', 'lbl', 'subject_id'])
    valData_name               = 'csv_v2/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)















# Nodules (Lungs)
disease_name       = 'NodulesPatient'
folds              = 5
patch_folder       = '/DataFolder/lungs/final_dataset/complete/NIfTI/train-val/Nodules/'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    
    # Extracting Names
    subject_id           = patient_list[i].split('.nii.gz')[0].split('CT_')[1]
    subject_id_list.append(subject_id)
    # Setting I/O Path
    patient_path_list.append(patch_folder + patient_list[i])

# Synchronous Data Shuffle
a, b, c  = shuffle(patient_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
columns           = ['img', 'lbl','subject_id'])
img               = shuffled_dataset['img']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = lbl[train]
    c_train = subject_id[train]
    a_val   = img[val]
    b_val   = lbl[val]
    c_val   = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['img', 'lbl', 'subject_id'])
    trainData_name             = 'csv_v2/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['img', 'lbl', 'subject_id'])
    valData_name               = 'csv_v2/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)

















# Pneumonia-Atelectasis (Lungs)
disease_name       = 'Pneumonia-AtelectasisPatient'
folds              = 5
patch_folder       = '/DataFolder/lungs/final_dataset/complete/NIfTI/train-val/Pneumonia-Atelectasis/'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    
    # Extracting Names
    subject_id           = patient_list[i].split('.nii.gz')[0].split('CT_')[1]
    subject_id_list.append(subject_id)
    # Setting I/O Path
    patient_path_list.append(patch_folder + patient_list[i])

# Synchronous Data Shuffle
a, b, c  = shuffle(patient_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
columns           = ['img', 'lbl','subject_id'])
img               = shuffled_dataset['img']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = lbl[train]
    c_train = subject_id[train]
    a_val   = img[val]
    b_val   = lbl[val]
    c_val   = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['img', 'lbl', 'subject_id'])
    trainData_name             = 'csv_v2/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['img', 'lbl', 'subject_id'])
    valData_name               = 'csv_v2/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)



















# Mass (Lungs)
disease_name       = 'MassPatient'
folds              = 5
patch_folder       = '/DataFolder/lungs/final_dataset/complete/NIfTI/train-val/Mass/'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    
    # Extracting Names
    subject_id           = patient_list[i].split('.nii.gz')[0].split('CT_')[1]
    subject_id_list.append(subject_id)
    # Setting I/O Path
    patient_path_list.append(patch_folder + patient_list[i])

# Synchronous Data Shuffle
a, b, c  = shuffle(patient_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
columns           = ['img', 'lbl','subject_id'])
img               = shuffled_dataset['img']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = lbl[train]
    c_train = subject_id[train]
    a_val   = img[val]
    b_val   = lbl[val]
    c_val   = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['img', 'lbl', 'subject_id'])
    trainData_name             = 'csv_v2/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['img', 'lbl', 'subject_id'])
    valData_name               = 'csv_v2/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)























# MultipleDiseases (Lungs)
disease_name       = 'MultipleDiseasesPatient'
folds              = 5
patch_folder       = '/DataFolder/lungs/final_dataset/complete/NIfTI/train-val/MultipleDiseases/'
label_sign         = 1

patient_list       = os.listdir(patch_folder)
patient_path_list  = []
subject_id_list    = []

# Populating Lists
class_label = np.full(len(patient_list), label_sign)
for i in range(0,len(patient_list)):
    
    # Extracting Names
    subject_id           = patient_list[i].split('.nii.gz')[0].split('CT_')[1]
    subject_id_list.append(subject_id)
    # Setting I/O Path
    patient_path_list.append(patch_folder + patient_list[i])

# Synchronous Data Shuffle
a, b, c  = shuffle(patient_path_list, class_label, subject_id_list, random_state=2)

# Metadata Setup
shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
columns           = ['img', 'lbl','subject_id'])
img               = shuffled_dataset['img']
lbl               = shuffled_dataset['lbl']
subject_id        = shuffled_dataset['subject_id']

# Generating CSV
kf     = KFold(folds)
fold   = 0
for train, val in kf.split(a):
    fold +=1
    print(disease_name + "; Fold #" + str(fold))

    a_train = img[train]
    b_train = lbl[train]
    c_train = subject_id[train]
    a_val   = img[val]
    b_val   = lbl[val]
    c_val   = subject_id[val]

    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['img', 'lbl', 'subject_id'])
    trainData_name             = 'csv_v2/{}-Training-Fold-{}'.format(disease_name, fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['img', 'lbl', 'subject_id'])
    valData_name               = 'csv_v2/{}-Validation-Fold-{}'.format(disease_name, fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)