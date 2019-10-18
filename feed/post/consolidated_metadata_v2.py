import SimpleITK as sitk
import os
import pandas as pd
import numpy as np
import glob
import os
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from sklearn.utils import shuffle
from sklearn.model_selection import KFold



'''
 Binary Classification (3D)
 Feed: Consolidating Metadata

 Update: 16/10/2019
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

OptIO  = '/SSD0/lungs/numpy/'
folds  = 5
fold   = 0

for im in range(0,folds):
    fold += 1
    
    # Training Data
    multiclassTrainName     = 'Lungs_CV'
    load_normal_fold        = pd.read_csv("csv_v2/NormalLungsPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease1_fold      = pd.read_csv("csv_v2/EmphysemaPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease2_fold      = pd.read_csv("csv_v2/Pneumonia-AtelectasisPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease3_fold      = pd.read_csv("csv_v2/NodulesPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease4_fold      = pd.read_csv("csv_v2/MassPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease5_fold      = pd.read_csv("csv_v2/MultipleDiseasesPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    train_fold_img          = []
    train_fold_lbl          = []
    train_fold_subject_id   = []

    # Normal
    normal_lbl         = load_normal_fold['lbl']
    normal_subject_id  = load_normal_fold['subject_id']
    for i in range (0,len(normal_subject_id)):
        img          =  OptIO + 'Normal/CT_' + normal_subject_id[i] + '.npy' 
        lbl          =  normal_lbl[i]        
        subject_id   =  normal_subject_id[i] 
        train_fold_img.append(img)
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id)  
    print('Loading Normal Training Metadata:' + str(len(subject_id)))


    # Disease 1
    disease1_lbl         = load_disease1_fold['lbl']
    disease1_subject_id  = load_disease1_fold['subject_id']
    for i in range (0,len(disease1_subject_id)):
        img          =  OptIO + 'Emphysema/CT_' + disease1_subject_id[i] + '.npy'
        lbl          =  disease1_lbl[i]        
        subject_id   =  disease1_subject_id[i]
        train_fold_img.append(img) 
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id) 
    print('Loading Emphysema Training Metadata:' + str(len(subject_id)))


    # Disease 2
    disease2_lbl         = load_disease2_fold['lbl']
    disease2_subject_id  = load_disease2_fold['subject_id']
    for i in range (0,len(disease2_subject_id)):
        img          =  OptIO + 'Pneumonia-Atelectasis/CT_' + disease2_subject_id[i] + '.npy'
        lbl          =  disease2_lbl[i]        
        subject_id   =  disease2_subject_id[i]
        train_fold_img.append(img) 
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id) 
    print('Loading Pneumonia-Atelectasis Training Metadata:' + str(len(subject_id)))



    # Disease 3
    disease3_lbl         = load_disease3_fold['lbl']
    disease3_subject_id  = load_disease3_fold['subject_id']
    for i in range (0,len(disease3_subject_id)):
        img          =  OptIO + 'Nodules/CT_' + disease3_subject_id[i] + '.npy'
        lbl          =  disease3_lbl[i]        
        subject_id   =  disease3_subject_id[i] 
        train_fold_img.append(img) 
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id) 
    print('Loading Nodules Training Metadata:' + str(len(subject_id)))



    # Disease 4
    disease4_lbl         = load_disease4_fold['lbl']
    disease4_subject_id  = load_disease4_fold['subject_id']
    for i in range (0,len(disease4_subject_id)):
        img          =  OptIO + 'Mass/CT_' + disease4_subject_id[i] + '.npy'
        lbl          =  disease4_lbl[i]        
        subject_id   =  disease4_subject_id[i]
        train_fold_img.append(img) 
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id) 
    print('Loading Mass Training Metadata:' + str(len(subject_id)))



    # Disease 5
    disease5_lbl         = load_disease5_fold['lbl']
    disease5_subject_id  = load_disease5_fold['subject_id']
    for i in range (0,len(disease5_subject_id)):
        img          =  OptIO + 'MultipleDiseases/CT_' + disease5_subject_id[i] + '.npy'
        lbl          =  disease5_lbl[i]        
        subject_id   =  disease5_subject_id[i]
        train_fold_img.append(img) 
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id) 
    print('Loading Multiple Diseases Training Metadata:' + str(len(subject_id)))



    # Synchronous Data Shuffle
    a, b, c  =  shuffle(train_fold_img, train_fold_lbl, train_fold_subject_id, random_state=2)


    trainData      = pd.DataFrame(list(zip(a,b,c)),
    columns        = ['img', 'lbl', 'subject_id'])
    trainData_name ='csv_v2/{}-Training-Fold-{}'.format(multiclassTrainName,fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)






    # Validation Data
    multiclassTrainName     = 'Lungs_CV'
    load_normal_fold        = pd.read_csv("csv_v2/NormalLungsPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease1_fold      = pd.read_csv("csv_v2/EmphysemaPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease2_fold      = pd.read_csv("csv_v2/Pneumonia-AtelectasisPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease3_fold      = pd.read_csv("csv_v2/NodulesPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease4_fold      = pd.read_csv("csv_v2/MassPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease5_fold      = pd.read_csv("csv_v2/MultipleDiseasesPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    val_fold_img            = []
    val_fold_lbl            = []
    val_fold_subject_id     = []

    # Normal
    normal_lbl         = load_normal_fold['lbl']
    normal_subject_id  = load_normal_fold['subject_id']
    for i in range (0,len(normal_subject_id)):
        img          =  OptIO + 'Normal/CT_' + normal_subject_id[i] + '.npy'
        lbl          =  normal_lbl[i]        
        subject_id   =  normal_subject_id[i] 
        val_fold_img.append(img)
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id)  
    print('Loading Normal Validation Metadata:' + str(len(subject_id)))


    # Disease 1
    disease1_lbl         = load_disease1_fold['lbl']
    disease1_subject_id  = load_disease1_fold['subject_id']
    for i in range (0,len(disease1_subject_id)):
        img          =  OptIO + 'Emphysema/CT_' + disease1_subject_id[i] + '.npy'
        lbl          =  disease1_lbl[i]        
        subject_id   =  disease1_subject_id[i]
        val_fold_img.append(img) 
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id) 
    print('Loading Emphysema Validation Metadata:' + str(len(subject_id)))


    # Disease 2
    disease2_lbl         = load_disease2_fold['lbl']
    disease2_subject_id  = load_disease2_fold['subject_id']
    for i in range (0,len(disease2_subject_id)):
        img          =  OptIO + 'Pneumonia-Atelectasis/CT_' + disease2_subject_id[i] + '.npy'
        lbl          =  disease2_lbl[i]        
        subject_id   =  disease2_subject_id[i]
        val_fold_img.append(img) 
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id) 
    print('Loading Pneumonia-Atelectasis Validation Metadata:' + str(len(subject_id)))



    # Disease 3
    disease3_lbl         = load_disease3_fold['lbl']
    disease3_subject_id  = load_disease3_fold['subject_id']
    for i in range (0,len(disease3_subject_id)):
        img          =  OptIO + 'Nodules/CT_' + disease3_subject_id[i] + '.npy'
        lbl          =  disease3_lbl[i]        
        subject_id   =  disease3_subject_id[i] 
        val_fold_img.append(img) 
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id) 
    print('Loading Nodules Validation Metadata:' + str(len(subject_id)))



    # Disease 4
    disease4_lbl         = load_disease4_fold['lbl']
    disease4_subject_id  = load_disease4_fold['subject_id']
    for i in range (0,len(disease4_subject_id)):
        img          =  OptIO + 'Mass/CT_' + disease4_subject_id[i] + '.npy'
        lbl          =  disease4_lbl[i]        
        subject_id   =  disease4_subject_id[i]
        val_fold_img.append(img) 
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id) 
    print('Loading Mass Validation Metadata:' + str(len(subject_id)))



    # Disease 5
    disease5_lbl         = load_disease5_fold['lbl']
    disease5_subject_id  = load_disease5_fold['subject_id']
    for i in range (0,len(disease5_subject_id)):
        img          =  OptIO + 'MultipleDiseases/CT_' + disease5_subject_id[i] + '.npy'
        lbl          =  disease5_lbl[i]        
        subject_id   =  disease5_subject_id[i]
        val_fold_img.append(img) 
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id) 
    print('Loading Multiple Diseases Validation Metadata:' + str(len(subject_id)))



    # Synchronous Data Shuffle
    a, b, c  =  shuffle(val_fold_img, val_fold_lbl, val_fold_subject_id, random_state=2)


    valData        = pd.DataFrame(list(zip(a,b,c)),
    columns        = ['img', 'lbl', 'subject_id'])
    valData_name   ='csv_v2/{}-Validation-Fold-{}'.format(multiclassTrainName,fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)
