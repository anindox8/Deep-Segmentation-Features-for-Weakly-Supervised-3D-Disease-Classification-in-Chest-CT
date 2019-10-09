import numpy as np
import os
import pandas as pd
import dicom2nifti


raw_data_path  =  '/DataFolder/lungs/Prior_821/original_volumes/Collective/'
raw_list       =   os.listdir(raw_data_path)
raw_save_path  =  '/DataFolder/lungs/Current_1445/'
counter        =   0

# Patient-Level Directory
for scan in range(0,len(raw_list)):
    scan_path         = raw_data_path + raw_list[scan]
    accession         = raw_list[scan].split('.nii.gz')[0].split('CT_')[1].split('_')[0]

    # Read Disease Checklist
    tusharlist  = pd.read_csv("Tushar-Case_Checklist.csv",dtype=object,keep_default_na=False,na_values=[])
    reportCSV   = tusharlist['Fake_Accession_Number_(Subject)'].tolist()
    labelCSV    = tusharlist['Label']
    patientCSV  = tusharlist['Fake_ID']

    try:
        # Retrieve Disease Label
        report_index  = reportCSV.index(str(accession))
        label         = labelCSV[report_index]
        patient_name  = patientCSV[report_index]
    
        save_path     = raw_save_path + label + '/CT_' + patient_name.split('CT')[1] + '_' + raw_list[scan].split('CT_')[1]
    
        os.system('cp {} {}'.format(scan_path, save_path))
    except:
        counter = counter + 1