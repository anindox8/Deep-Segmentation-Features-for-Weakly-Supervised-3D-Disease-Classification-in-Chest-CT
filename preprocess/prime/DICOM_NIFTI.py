import numpy as np
import os
import pandas as pd
import dicom2nifti



'''
 3D Binary Classification
 DICOM-NIFTI Conversion and Folder Sorting
 As Per Disease Labeling

 Update: 12/08/2019
 Contributors: as1044, ft42
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Emphysema
     - Pneumonia-Atelectasis
     - Mass
     - Nodules

'''


raw_data_path  =  '/DataFolder/rh163/CT_Batch_Deidentify/Deidentify_CT_Case298_Batch16_Aug10/'
raw_list       =   os.listdir(raw_data_path)
raw_save_path  =  '/DataFolder/as1044/NIFTI_01/'


patient_DICOM_name_list           =   []
patient_report_number_list        =   []
patient_scan_number_list          =   []
patient_scan_path_list            =   []
nifty_path                        =   []

DICOM2NIFTI_problem_cases_ID      =   []
DICOM2NIFTI_problem_cases_SCAN    =   []
DICOM2NIFTI_problem_cases_REPORT  =   []


# Patient-Level Directory
for patient in range(0,len(raw_list)):

    if ((raw_list[patient]=='Exceptions')|(raw_list[patient]=='NoPHI_logfile_processed_extra_studies_Case298_Batch16_Aug10.csv')):
        print('Skipping:' + raw_list[patient])
        continue
    else:
        patient_path         = raw_data_path + raw_list[patient] + '/'
        patient_report_list  = [dI for dI in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path,dI))]
    
        # Report-Level Directory
        for patient_reports in range(0,len(patient_report_list)):
            report_path          = patient_path + patient_report_list[patient_reports] + '/'
            patient_scans_list   = os.listdir(report_path)
            
            scans_length         = []
            scans_length_path    = []
            
            # Scan-Level Directory
            for l in range(0,len(patient_scans_list)):
                scans_path       = report_path + patient_scans_list[l]+'/'
                length           = len(os.listdir(scans_path))
                scans_length.append(length)
                scans_length_path.append(scans_path)
    
            if scans_length:
                scan_with_maximum_number_of_slices = np.argmax(scans_length)
                scans_path                         = scans_length_path[scan_with_maximum_number_of_slices]
                name                               = 'CT_' + raw_list[patient].split('CT')[1] + '_' + patient_report_list[patient_reports] + '_' + patient_scans_list[scan_with_maximum_number_of_slices] + '.nii.gz'
                print('Serial-{};  PatientID:{};  Report:{};  Scan:{}.'.format(patient,raw_list[patient].split('CT')[1],patient_report_list[patient_reports],patient_scans_list[scan_with_maximum_number_of_slices]))        
            else:
                name                               = 'CT_' + raw_list[patient].split('CT')[1] + '_' + 'VERIFY' + '_' + 'EMPTY.nii.gz'
                print('Serial-{};  PatientID:{};  Report:{};  Scan:EMPTY.'.format(patient,raw_list[patient].split('CT')[1],patient_report_list[patient_reports]))
    
        
        
            # Converting to NIFTI
            try:
    
                # Read Disease Checklist
                diseaselist = pd.read_csv("Disease-Case_Checklist.csv",dtype=object,keep_default_na=False,na_values=[])
                reportCSV   = diseaselist['Fake_Accession_Number_(Subject)'].tolist()
                label0CSV   = diseaselist['Label0']
                label1CSV   = diseaselist['Label1']
                
                # Retrieve Disease Label
                report_index  = reportCSV.index(str(patient_report_list[patient_reports]))
                diseaselabel0 = label0CSV[report_index]
                diseaselabel1 = label1CSV[report_index]
                
                # Multiple Diseases
                if (len(diseaselabel1)>3):
                    print('Labels: {}; {}'.format(diseaselabel0, diseaselabel1))
                    output_file = raw_save_path + 'MultipleDiseases/' + name
                    if not os.path.exists(output_file):   # Disable Overwriting
                        dicom2nifti.dicom_series_to_nifti(scans_path, output_file, reorient_nifti=True)
                
                # Single Disease
                else:
                    print('Label: {}'.format(diseaselabel0))
                    output_file = raw_save_path + diseaselabel0.strip() + '/' + name
                    if not os.path.exists(output_file):   # Disable Overwriting
                        dicom2nifti.dicom_series_to_nifti(scans_path, output_file, reorient_nifti=True)
                    
                
                patient_DICOM_name_list.append(raw_list[patient])
                patient_report_number_list.append(patient_report_list[patient_reports])
                patient_scan_number_list.append(patient_scans_list[scan_with_maximum_number_of_slices])
                patient_scan_path_list.append(scans_path)
                nifty_path.append(output_file)
                        
            except:
                
                print('ERROR!')
    
                DICOM2NIFTI_problem_cases_ID.append(raw_list[patient])
                if scans_length:
                    DICOM2NIFTI_problem_cases_SCAN.append(patient_scans_list[scan_with_maximum_number_of_slices])
                    DICOM2NIFTI_problem_cases_REPORT.append(patient_report_list[patient_reports])
                else:
                    DICOM2NIFTI_problem_cases_SCAN.append(9999999)   # Error Flag Code: Empty Directory
                    DICOM2NIFTI_problem_cases_REPORT.append(88888888)
                pass


success_data = pd.DataFrame(list(zip(patient_DICOM_name_list, patient_report_number_list, patient_scan_number_list, patient_scan_path_list, nifty_path)),
                        columns=['ID','Report','Scan','Path','NIFTI_Path'])
success_data.to_csv("SuccessDICOM_NIFTI.csv", encoding='utf-8', index=False)

error_data   = pd.DataFrame(list(zip(DICOM2NIFTI_problem_cases_ID, DICOM2NIFTI_problem_cases_REPORT, DICOM2NIFTI_problem_cases_SCAN)),
                        columns=['ID','Report','Scan',])
error_data.to_csv("ErrorDICOM_NIFTI.csv", encoding='utf-8', index=False)

