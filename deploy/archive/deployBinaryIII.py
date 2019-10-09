from __future__ import division
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import predictor
from dltk.io.augmentation import extract_random_example_array
from readerIII import read_fn



'''
 3D Binary Classification
 Train: Preprocessing Volumes (Padding + Normalization)

 Update: 30/07/2019
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

PATCH         = 112

READER_PARAMS = {'extract_patches':          False,
                 'augmentation':             False,
                 'n_patches':                    1,
                 'patch_size':[PATCH, PATCH, PATCH]}


def predict(args):
    
    id_list              =  []     # Patient ID
    probability_list     =  []     # Probabilities
    label_list           =  []     # Labels
    class_1_list         =  []     # Class 1
    class_2_list         =  []     # Class 2


   # Read CSV with Validation/Test Set
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).values

    # Load Trained Model
    export_dir = \
        [os.path.join(args.model_path, o) for o in sorted(os.listdir(args.model_path))
         if os.path.isdir(os.path.join(args.model_path, o)) and o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through Files, Predict on the Full Volumes, Compute Dice
    accuracy = []
    for output in read_fn(file_references = file_names,
                          mode            = tf.estimator.ModeKeys.PREDICT,
                          params          = READER_PARAMS):
        
        t0 = time.time()  # Timing Function

        # Parse Data Reader Output
        img      = output['features']['x']
        lbl      = output['labels']['y']
        test_id  = output['img_id']

        # Decompose Volumes into Patches        
        num_crop_predictions = 6
        crop_batch = extract_random_example_array(
            image_list     = img,
            example_size   = [PATCH, PATCH, PATCH],
            n_examples     = num_crop_predictions)

        # Generate Predictions
        y_ = my_predictor.session.run(
            fetches    =  my_predictor._fetch_tensors['y_prob'],
            feed_dict  = {my_predictor._feed_tensors['x']: crop_batch})

        # Average Predictions on Cropped Test Input:
        y_ = np.mean(y_, axis=0)
        predicted_class = np.argmax(y_) 

        # Populate Lists
        id_list.append(test_id)
        probability_list.append(y_)
        label_list.append(lbl[0])
        class_1_list.append(y_[0])
        class_2_list.append(y_[1])


        # Print Outputs
        print('ID={}; Prediction={}; True={}; AvgProb={}; Run Time={:0.2f} s; '
              ''.format(test_id, predicted_class, lbl[0], y_, time.time()-t0))


    deployclf_data = pd.DataFrame(list(zip(id_list,probability_list,label_list,class_1_list,class_2_list)),
    columns=['id','prob','y_true','class0','class1'])
    deployclf_data.to_csv("ConnectedLungsBinary3D_Fold-1_150K_StepsO.csv", encoding='utf-8', index=False)


if __name__ == '__main__':

    # Argument Parser Setup
    parser = argparse.ArgumentParser(description='Binary Lungs Disease Classification')
    parser.add_argument('--verbose',            default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path',   '-m', default='/Local/scripts/lungs/classification/weights/29072019/Fold_1Step5000valloss0.68635')
    parser.add_argument('--csv',          '-d', default='/Local/scripts/lungs/classification/feed/csvIII/Lung_CV-Validation-Fold-1.csv')

    args = parser.parse_args()

    # Set Verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU Allocation Options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Allow GPU Usage Growth
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Inference
    predict(args)

    session.close()


