import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


'''
 Binary Classification (3D)
 Feed: Generating Metadata

 Update: 27/08/2019
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


# Feeding Script Input Parameters
organ_name          =  'Lungs'
disease_names       = ['Normal', 'Emphysema', 'Nodules', 'Pneumonia-Atelectasis', 'Mass', 'MultipleDiseases']
label_signs         = [ 0, 1, 1, 1, 1, 1 ]
partitions          = ['Training-Fold', 'Validation-Fold']
save_path           =  'csvIV'
folds               =   4
data_subset         =  'trainY/'
HDF5_path           =  '/DataFolder/lungs/Final_1593/HDF5/'





# Generating Feed Directories for Class Folds
for D in range(len(disease_names)):

    disease_name       = disease_names[D]
    label_sign         = label_signs[D]
    patch_folder       = '/DataFolder/lungs/Final_1593/train01/' + disease_name + '/'

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
        trainData_name             = 'csvIV/{}_Training-Fold-{}'.format(disease_name, fold)+'.csv'
        trainData.to_csv(trainData_name, encoding='utf-8', index=False)
    
        valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
        columns                    = ['img', 'lbl', 'subject_id'])
        valData_name               = save_path + '/{}_Validation-Fold-{}'.format(disease_name, fold)+'.csv'
        valData.to_csv(valData_name, encoding='utf-8', index=False)





# Consolidating Feed Directories for Complete Folds
for P in range(len(partitions)):
    for F in range(0,folds):
        
        # Training Data    
        load_normal_fold        = pd.read_csv(save_path + "/" + disease_names[0] + "_" + partitions[P] + "-{}.csv".format(F+1),dtype=object,keep_default_na=False,na_values=[])
        load_disease1_fold      = pd.read_csv(save_path + "/" + disease_names[1] + "_" + partitions[P] + "-{}.csv".format(F+1),dtype=object,keep_default_na=False,na_values=[])
        load_disease2_fold      = pd.read_csv(save_path + "/" + disease_names[2] + "_" + partitions[P] + "-{}.csv".format(F+1),dtype=object,keep_default_na=False,na_values=[])
        load_disease3_fold      = pd.read_csv(save_path + "/" + disease_names[3] + "_" + partitions[P] + "-{}.csv".format(F+1),dtype=object,keep_default_na=False,na_values=[])
        load_disease4_fold      = pd.read_csv(save_path + "/" + disease_names[4] + "_" + partitions[P] + "-{}.csv".format(F+1),dtype=object,keep_default_na=False,na_values=[])
        load_disease5_fold      = pd.read_csv(save_path + "/" + disease_names[5] + "_" + partitions[P] + "-{}.csv".format(F+1),dtype=object,keep_default_na=False,na_values=[])
        
        train_fold_img          = []
        train_fold_lbl          = []
        train_fold_subject_id   = []
    
        # Normal
        normal_lbl         = load_normal_fold['lbl']
        normal_subject_id  = load_normal_fold['subject_id']
        for i in range (0,len(normal_subject_id)):
            img          =  HDF5_path + data_subset + disease_names[0] + '.h5'
            lbl          =  normal_lbl[i]        
            subject_id   =  normal_subject_id[i] 
            train_fold_img.append(img)
            train_fold_lbl.append(lbl)        
            train_fold_subject_id.append(subject_id)  
        print('Loading ' + disease_names[0] + ' ' + partitions[P] + ' Metadata: ' + str(len(subject_id)))
    
    
        # Disease 1
        disease1_lbl         = load_disease1_fold['lbl']
        disease1_subject_id  = load_disease1_fold['subject_id']
        for i in range (0,len(disease1_subject_id)):
            img          =  HDF5_path + data_subset + disease_names[1] + '.h5'
            lbl          =  disease1_lbl[i]        
            subject_id   =  disease1_subject_id[i]
            train_fold_img.append(img) 
            train_fold_lbl.append(lbl)        
            train_fold_subject_id.append(subject_id) 
        print('Loading ' + disease_names[1] + ' ' + partitions[P] + ' Metadata: ' + str(len(subject_id)))
    
    
        # Disease 2
        disease2_lbl         = load_disease2_fold['lbl']
        disease2_subject_id  = load_disease2_fold['subject_id']
        for i in range (0,len(disease2_subject_id)):
            img          =  HDF5_path + data_subset + disease_names[2] + '.h5'
            lbl          =  disease2_lbl[i]        
            subject_id   =  disease2_subject_id[i]
            train_fold_img.append(img) 
            train_fold_lbl.append(lbl)        
            train_fold_subject_id.append(subject_id) 
        print('Loading ' + disease_names[2] + ' ' + partitions[P] + ' Metadata: ' + str(len(subject_id)))
    
    
    
        # Disease 3
        disease3_lbl         = load_disease3_fold['lbl']
        disease3_subject_id  = load_disease3_fold['subject_id']
        for i in range (0,len(disease3_subject_id)):
            img          =  HDF5_path + data_subset + disease_names[3] + '.h5'
            lbl          =  disease3_lbl[i]        
            subject_id   =  disease3_subject_id[i] 
            train_fold_img.append(img) 
            train_fold_lbl.append(lbl)        
            train_fold_subject_id.append(subject_id) 
        print('Loading ' + disease_names[3] + ' ' + partitions[P] + ' Metadata: ' + str(len(subject_id)))
    
    
    
        # Disease 4
        disease4_lbl         = load_disease4_fold['lbl']
        disease4_subject_id  = load_disease4_fold['subject_id']
        for i in range (0,len(disease4_subject_id)):
            img          =  HDF5_path + data_subset + disease_names[4] + '.h5'
            lbl          =  disease4_lbl[i]        
            subject_id   =  disease4_subject_id[i]
            train_fold_img.append(img) 
            train_fold_lbl.append(lbl)        
            train_fold_subject_id.append(subject_id) 
        print('Loading ' + disease_names[4] + ' ' + partitions[P] + ' Metadata: ' + str(len(subject_id)))
    
    
    
        # Disease 5
        disease5_lbl         = load_disease5_fold['lbl']
        disease5_subject_id  = load_disease5_fold['subject_id']
        for i in range (0,len(disease5_subject_id)):
            img          =  HDF5_path + data_subset + disease_names[5] + '.h5'
            lbl          =  disease5_lbl[i]        
            subject_id   =  disease5_subject_id[i]
            train_fold_img.append(img) 
            train_fold_lbl.append(lbl)        
            train_fold_subject_id.append(subject_id) 
        print('Loading ' + disease_names[5] + ' ' + partitions[P] + ' Metadata: ' + str(len(subject_id)))
    
    
    
        # Synchronous Data Shuffle
        a, b, c  =  shuffle(train_fold_img, train_fold_lbl, train_fold_subject_id, random_state=2)
    
    
        trainData      =  pd.DataFrame(list(zip(a,b,c)),
        columns        =  ['img', 'lbl', 'subject_id'])
        trainData_name =  save_path + '/' + organ_name + '_' + partitions[P] + '-{}.csv'.format(F+1)
        trainData.to_csv(trainData_name, encoding='utf-8', index=False)    