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



folds = 4
fold  = 0

for im in range(0,folds):
    fold += 1
    
    # Training Data
    multiclassTrainName ='Lung_CV'
    load_normal_fold        = pd.read_csv("csvIII/NormalLungPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease1_fold      = pd.read_csv("csvIII/EdemaPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease2_fold      = pd.read_csv("csvIII/AtelectasisPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease3_fold      = pd.read_csv("csvIII/PneumoniaPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease4_fold      = pd.read_csv("csvIII/NodulesPatient-Training-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    train_fold_img          = []
    train_fold_binary_mask  = []
    train_fold_feature_map  = []
    train_fold_lbl          = []
    train_fold_subject_id   = []


    # Normal
    normal_img         = load_normal_fold['img']
    normal_binary_mask = load_normal_fold['binary_mask']
    normal_feature_map = load_normal_fold['feature_map']
    normal_lbl         = load_normal_fold['lbl']
    normal_subject_id  = load_normal_fold['subject_id']

    for i in range (0,len(normal_img)):

        img          =  normal_img[i]        
        binary_mask  =  normal_binary_mask[i]
        feature_map  =  normal_feature_map[i]
        lbl          =  normal_lbl[i]        
        subject_id   =  normal_subject_id[i] 

        train_fold_img.append(img)        
        train_fold_binary_mask.append(binary_mask)
        train_fold_feature_map.append(feature_map)
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id) 
    
    print('Loading Normal Training Metadata:' + str(len(train_fold_img)))



    # Disease 1
    disease1_img         = load_disease1_fold['img']
    disease1_binary_mask = load_disease1_fold['binary_mask']
    disease1_feature_map = load_disease1_fold['feature_map']
    disease1_lbl         = load_disease1_fold['lbl']
    disease1_subject_id  = load_disease1_fold['subject_id']

    for i in range (0,len(disease1_img)):

        img          =  disease1_img[i]        
        binary_mask  =  disease1_binary_mask[i]
        feature_map  =  disease1_feature_map[i]
        lbl          =  disease1_lbl[i]        
        subject_id   =  disease1_subject_id[i] 

        train_fold_img.append(img)        
        train_fold_binary_mask.append(binary_mask)
        train_fold_feature_map.append(feature_map)
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id)
    
    print('Loading Edema Training Metadata:' + str(len(train_fold_img)))



    # Disease 2
    disease2_img         = load_disease2_fold['img']
    disease2_binary_mask = load_disease2_fold['binary_mask']
    disease2_feature_map = load_disease2_fold['feature_map']
    disease2_lbl         = load_disease2_fold['lbl']
    disease2_subject_id  = load_disease2_fold['subject_id']

    for i in range (0,len(disease2_img)):

        img          =  disease2_img[i]        
        binary_mask  =  disease2_binary_mask[i]
        feature_map  =  disease2_feature_map[i]
        lbl          =  disease2_lbl[i]        
        subject_id   =  disease2_subject_id[i] 

        train_fold_img.append(img)        
        train_fold_binary_mask.append(binary_mask)
        train_fold_feature_map.append(feature_map)
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id)
    
    print('Loading Atelectasis Training Metadata:' + str(len(train_fold_img)))



    # Disease 3
    disease3_img         = load_disease3_fold['img']
    disease3_binary_mask = load_disease3_fold['binary_mask']
    disease3_feature_map = load_disease3_fold['feature_map']
    disease3_lbl         = load_disease3_fold['lbl']
    disease3_subject_id  = load_disease3_fold['subject_id']

    for i in range (0,len(disease3_img)):

        img          =  disease3_img[i]        
        binary_mask  =  disease3_binary_mask[i]
        feature_map  =  disease3_feature_map[i]
        lbl          =  disease3_lbl[i]        
        subject_id   =  disease3_subject_id[i] 

        train_fold_img.append(img)        
        train_fold_binary_mask.append(binary_mask)
        train_fold_feature_map.append(feature_map)
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id)
    
    print('Loading Pneumonia Training Metadata:' + str(len(train_fold_img)))




    # Disease 4
    disease4_img         = load_disease4_fold['img']
    disease4_binary_mask = load_disease4_fold['binary_mask']
    disease4_feature_map = load_disease4_fold['feature_map']
    disease4_lbl         = load_disease4_fold['lbl']
    disease4_subject_id  = load_disease4_fold['subject_id']

    for i in range (0,len(disease4_img)):

        img          =  disease4_img[i]        
        binary_mask  =  disease4_binary_mask[i]
        feature_map  =  disease4_feature_map[i]
        lbl          =  disease4_lbl[i]        
        subject_id   =  disease4_subject_id[i] 

        train_fold_img.append(img)        
        train_fold_binary_mask.append(binary_mask)
        train_fold_feature_map.append(feature_map)
        train_fold_lbl.append(lbl)        
        train_fold_subject_id.append(subject_id)
    
    print('Loading Nodules Training Metadata:' + str(len(train_fold_img)))






    # Synchronous Data Shuffle
    a, b, c, d, e  =  shuffle(train_fold_img, train_fold_binary_mask, train_fold_feature_map, train_fold_lbl, train_fold_subject_id, random_state=2)


    trainData      = pd.DataFrame(list(zip(a,b,c,d,e)),
    columns        = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    trainData_name ='csvIII/{}-Training-Fold-{}'.format(multiclassTrainName,fold)+'.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

   









    # Validation Data
    multiclassValName ='Lung_CV'
    load_normal_fold        = pd.read_csv("csvIII/NormalLungPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease1_fold      = pd.read_csv("csvIII/EdemaPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease2_fold      = pd.read_csv("csvIII/AtelectasisPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease3_fold      = pd.read_csv("csvIII/PneumoniaPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    load_disease4_fold      = pd.read_csv("csvIII/NodulesPatient-Validation-Fold-{}.csv".format(fold),dtype=object,keep_default_na=False,na_values=[])
    val_fold_img          = []
    val_fold_binary_mask  = []
    val_fold_feature_map  = []
    val_fold_lbl          = []
    val_fold_subject_id   = []


    # Normal
    normal_img         = load_normal_fold['img']
    normal_binary_mask = load_normal_fold['binary_mask']
    normal_feature_map = load_normal_fold['feature_map']
    normal_lbl         = load_normal_fold['lbl']
    normal_subject_id  = load_normal_fold['subject_id']

    for i in range (0,len(normal_img)):

        img          =  normal_img[i]        
        binary_mask  =  normal_binary_mask[i]
        feature_map  =  normal_feature_map[i]
        lbl          =  normal_lbl[i]        
        subject_id   =  normal_subject_id[i] 

        val_fold_img.append(img)        
        val_fold_binary_mask.append(binary_mask)
        val_fold_feature_map.append(feature_map)
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id) 
    
    print('Loading Normal Validation Metadata:' + str(len(val_fold_img)))



    # Disease 1
    disease1_img         = load_disease1_fold['img']
    disease1_binary_mask = load_disease1_fold['binary_mask']
    disease1_feature_map = load_disease1_fold['feature_map']
    disease1_lbl         = load_disease1_fold['lbl']
    disease1_subject_id  = load_disease1_fold['subject_id']

    for i in range (0,len(disease1_img)):

        img          =  disease1_img[i]        
        binary_mask  =  disease1_binary_mask[i]
        feature_map  =  disease1_feature_map[i]
        lbl          =  disease1_lbl[i]        
        subject_id   =  disease1_subject_id[i] 

        val_fold_img.append(img)        
        val_fold_binary_mask.append(binary_mask)
        val_fold_feature_map.append(feature_map)
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id)
    
    print('Loading Edema Validation Metadata:' + str(len(val_fold_img)))



    # Disease 2
    disease2_img         = load_disease2_fold['img']
    disease2_binary_mask = load_disease2_fold['binary_mask']
    disease2_feature_map = load_disease2_fold['feature_map']
    disease2_lbl         = load_disease2_fold['lbl']
    disease2_subject_id  = load_disease2_fold['subject_id']

    for i in range (0,len(disease2_img)):

        img          =  disease2_img[i]        
        binary_mask  =  disease2_binary_mask[i]
        feature_map  =  disease2_feature_map[i]
        lbl          =  disease2_lbl[i]        
        subject_id   =  disease2_subject_id[i] 

        val_fold_img.append(img)        
        val_fold_binary_mask.append(binary_mask)
        val_fold_feature_map.append(feature_map)
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id)
    
    print('Loading Atelectasis Validation Metadata:' + str(len(val_fold_img)))



    # Disease 3
    disease3_img         = load_disease3_fold['img']
    disease3_binary_mask = load_disease3_fold['binary_mask']
    disease3_feature_map = load_disease3_fold['feature_map']
    disease3_lbl         = load_disease3_fold['lbl']
    disease3_subject_id  = load_disease3_fold['subject_id']

    for i in range (0,len(disease3_img)):

        img          =  disease3_img[i]        
        binary_mask  =  disease3_binary_mask[i]
        feature_map  =  disease3_feature_map[i]
        lbl          =  disease3_lbl[i]        
        subject_id   =  disease3_subject_id[i] 

        val_fold_img.append(img)        
        val_fold_binary_mask.append(binary_mask)
        val_fold_feature_map.append(feature_map)
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id)
    
    print('Loading Pneumonia Validation Metadata:' + str(len(val_fold_img)))




    # Disease 4
    disease4_img         = load_disease4_fold['img']
    disease4_binary_mask = load_disease4_fold['binary_mask']
    disease4_feature_map = load_disease4_fold['feature_map']
    disease4_lbl         = load_disease4_fold['lbl']
    disease4_subject_id  = load_disease4_fold['subject_id']

    for i in range (0,len(disease4_img)):

        img          =  disease4_img[i]        
        binary_mask  =  disease4_binary_mask[i] 
        feature_map  =  disease4_feature_map[i]
        lbl          =  disease4_lbl[i]        
        subject_id   =  disease4_subject_id[i] 

        val_fold_img.append(img)        
        val_fold_binary_mask.append(binary_mask)
        val_fold_feature_map.append(feature_map)
        val_fold_lbl.append(lbl)        
        val_fold_subject_id.append(subject_id)
    
    print('Loading Nodules Validation Metadata:' + str(len(val_fold_img)))






    # Synchronous Data Shuffle
    a, b, c, d, e  =  shuffle(val_fold_img, val_fold_binary_mask, val_fold_feature_map, val_fold_lbl, val_fold_subject_id, random_state=2)


    valData        = pd.DataFrame(list(zip(a,b,c,d,e)),
    columns        = ['img', 'binary_mask', 'feature_map', 'lbl', 'subject_id'])
    valData_name   ='csvIII/{}-Validation-Fold-{}'.format(multiclassValName,fold)+'.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)
