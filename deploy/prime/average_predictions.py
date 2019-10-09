import pandas as pd

'''
 3D Binary Classification
 Deploy: Average Patch Prediction Probabilities

 Update: 03/08/2019
 Contributors: ft42, as1044
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Edema
     - Atelectasis
     - Pneumonia
     - Nodules

YStep38700valloss0.00019
'''

prediction=pd.read_csv("./val_sheets/Final/Z2_ValSet110E.csv", dtype=object, keep_default_na=False,na_values=[]).values
# prediction=pd.read_csv("./test_sheets/Y_TestSet430EP.csv", dtype=object, keep_default_na=False,na_values=[]).values
                        

# List of All Patients
path            = prediction[0][0]
patiend_id_list = []

for i in range(0,len(prediction)):
    patient_id = prediction[i][0]
    patiend_id_list.append(patient_id)

# List of Unique Patients
unique_patient_list = []
for x in patiend_id_list:
    if x not in unique_patient_list:
        unique_patient_list.append(x)

# Average Predictions
avg_pred0_list   = []
avg_pred1_list   = []
y_true        = []
patient_name  = []
for un in range(0,len(unique_patient_list)):
    rio      = unique_patient_list[un]
    counter  = 0
    pred0    = 0
    pred1    = 0
    for i in range(0,len(prediction)):
        if (patiend_id_list[i]==rio):
            pred0   += float(prediction[i][3])
            pred1   += float(prediction[i][4])
            lbl      = prediction[i][2]
            counter += 1

    avg_pred0  =  pred0/counter
    avg_pred1  =  pred1/counter

    avg_pred0_list.append(avg_pred0)
    avg_pred1_list.append(avg_pred1)
    patient_name.append(unique_patient_list[un])
    y_true.append(lbl)

# Generate CSV
avgPred_data = pd.DataFrame(list(zip(avg_pred0_list, avg_pred1_list, y_true, patient_name)),
  columns = ['p0','p1','y_true','subj_id'])
avgPred_data.to_csv("./val_sheets/Final/Z2_ValSet110EC.csv", encoding='utf-8', index=False)
# avgPred_data.to_csv("./test_sheets/Y_TestSet550EPC.csv", encoding='utf-8', index=False)
