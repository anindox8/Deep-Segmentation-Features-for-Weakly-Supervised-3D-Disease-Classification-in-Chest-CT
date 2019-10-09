from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from dltk.networks.regression_classification.resnet import resnet_3d
from dltk.networks.segmentation.unet import residual_unet_3d
from dltk.core.activations import leaky_relu
from dltk.io.abstract_reader import Reader
from readerIIIh import read_fn
from residual_3DPNet import residual_3DPNet
import json
import warnings



'''
 3D Binary Classification
 Train: Training Connected Model

 Update: 31/07/2019
 Contributors: as1044, ft42
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Edema
     - Atelectasis
     - Pneumonia
     - Nodules

 Connection (Updates):
 -- inputs     =    segnet_output_ops['logits']                                           : Res3DUNet Segmentation Network Logits as i/p feed to Classification Network 
 -- inputs     =    tf.concat(values=[features['x'],segnet_output_ops['logits']], axis=4) : Res3DUNet Segmentation Network Logits <concat> Input Volume as i/p feed to Classification Network 
 -- 'residual_3DPNet'                                                                     : Consolidated Segmentation-Classification Network linked by Pre-Softmax Final Feature Maps 
 -- 'all_tensors', densevnet.py, niftynet                                                 : Multi-Res DenseVNet Segmentation Feature Maps Aggregated with Spatial Priori (Organ Probability Map) 
     output'--POST-PROCESS('probability')                                                  [Single Channel] <concat> Input Volume as i/p feed to Classification Network
'''


count_steps = []
count_loss  = []

EVAL_EVERY_N_STEPS  =  500
EVAL_STEPS          =  10

PATCH               =  112
NUM_CLASSES         =  2
NUM_CHANNELS        =  2
BATCH_SIZE          =  4
SHUFFLE_CACHE_SIZE  =  64
PREFETCH_CACHE_SIZE =  BATCH_SIZE*4
MAX_STEPS           =  150000

LR_INITIAL          =  0.001
LRDECAY_STEPS       =  6130    # 40 Epochs = 6130 Steps
LRDECAY_RATE        =  0.8     # Reduce LR by 20%



def lrelu(x):
    return leaky_relu(x, 0.1)

def model_fn(features, labels, mode, params):

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):

        # Model Definition (resnet_3D)
        model_output_ops = resnet_3d(
            inputs                   = features['x'],          # Input: Concatenated Patches (dimensions=128cube,channel=61-63; defined by 'patch_size')
            num_res_units            = 2,
            num_classes              = NUM_CLASSES,
            filters                  = (16,32, 64, 128, 256),
            strides                  = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            mode                     = mode,
            activation               = tf.nn.relu6,
            kernel_initializer       = tf.initializers.variance_scaling(distribution='uniform'),
            bias_initializer         = tf.zeros_initializer(),
            kernel_regularizer       = tf.contrib.layers.l2_regularizer(1e-3))
    
    
    
        # Prediction Mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode            = mode,
                predictions     = model_output_ops,
                export_outputs  = {'out': tf.estimator.export.PredictOutput(model_output_ops)})
    
        # Loss Function
        one_hot_labels = tf.reshape(tf.one_hot(labels['y'], depth=NUM_CLASSES), [-1, NUM_CLASSES])
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels      = one_hot_labels,
            logits             = model_output_ops['logits'])
    
        # Optimizer
        global_step = tf.train.get_global_step()
    
        # Learning Rate Decay (every 60 epochs with 0.2 decrease)
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        learning_rate = tf.train.exponential_decay(params["learning_rate"], global_step, decay_steps=LRDECAY_STEPS, decay_rate=LRDECAY_RATE, staircase=True)
        
        if params["opt"] == 'adam':
            optimiser = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=1e-5)
            optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)
        elif params["opt"] == 'momentum':
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
            optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)
        elif params["opt"] == 'rmsprop':
            optimiser = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate, momentum=0.9)
            optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)
    
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimiser.minimize(loss, global_step=global_step)
    
        # Custom Image Summaries (TensorBoard)
        my_image_summaries = {}
        my_image_summaries['CT_Patch'] = features['x'][0, 32, :, :, 0]
    
        expected_output_size = [1, PATCH, PATCH, 1]  # [B, W, H, C]
        [tf.summary.image(name, tf.reshape(image, expected_output_size))
         for name, image in my_image_summaries.items()]
    
        # Track Metrics
        acc             = tf.metrics.accuracy
        prec            = tf.metrics.precision
        auc             = tf.metrics.auc
        eval_metric_ops = {"accuracy":  acc(labels['y'],  model_output_ops['y_']),
                           "precision": prec(labels['y'], model_output_ops['y_']),
                           "auc":       prec(labels['y'], model_output_ops['y_'])}
    
        # Return EstimatorSpec Object
        return tf.estimator.EstimatorSpec(mode            = mode,
                                          predictions     = model_output_ops,
                                          loss            = loss,
                                          train_op        = train_op,
                                          eval_metric_ops = eval_metric_ops)


def train(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting Up...')

    # Import config.json Parameters
    with open(args.config) as f:
        run_config = json.load(f)

    # Read Training-Fold.csv
    train_filenames = pd.read_csv(
        args.train_csv, dtype=object, keep_default_na=False,
        na_values=[]).values

    # Read Validation-Fold.csv
    val_filenames = pd.read_csv(
        args.val_csv, dtype=object, keep_default_na=False,
        na_values=[]).values


    # Set DLTK Reader Parameters (No. of Patches, Patch Size) 
    reader_params = {'n_patches':        2,
                     'patch_size':       [PATCH, PATCH, PATCH],    # Target Patch Size
                     'extract_patches':  True,                     # Enable Training Mode Patch Extraction
                     'augmentation':     True}
    
    # Set Patch Dimensions
    reader_patch_shapes = {'features': {'x': reader_params['patch_size'] + [NUM_CHANNELS]},
                           'labels':   {'y': [1]}}
    
    # Initiate Data Reader + Patch Extraction
    reader = Reader(read_fn,
                  {'features': {'x': tf.float32},
                   'labels':   {'y': tf.int32}})


    # Create Input Functions + Queue Initialisation Hooks for Training/Validation Data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references       = train_filenames,
        mode                  = tf.estimator.ModeKeys.TRAIN,
        example_shapes        = reader_patch_shapes,
        batch_size            = BATCH_SIZE,
        shuffle_cache_size    = SHUFFLE_CACHE_SIZE,
        prefetch_cache_size   = PREFETCH_CACHE_SIZE,
        params                = reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references       = val_filenames,
        mode                  = tf.estimator.ModeKeys.EVAL,
        example_shapes        = reader_patch_shapes,
        batch_size            = BATCH_SIZE,
        shuffle_cache_size    = SHUFFLE_CACHE_SIZE,
        prefetch_cache_size   = PREFETCH_CACHE_SIZE,
        params                = reader_params)


    # Instantiate Neural Network Estimator
    nn = tf.estimator.Estimator(
        model_fn             = tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir            = args.model_path,
        params               = run_config,
        config               = tf.estimator.RunConfig())                                         


    # Hooks for Validation Summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(args.model_path, 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(every_n_steps = EVAL_EVERY_N_STEPS,
                                             output_dir    = args.model_path)


    print('Begin Training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn  = train_input_fn,
                hooks     = [train_qinit_hook, step_cnt_hook],
                steps     = EVAL_EVERY_N_STEPS)

            if args.run_validation:
                results_val   = nn.evaluate(
                    input_fn  = val_input_fn,
                    hooks     = [val_qinit_hook, val_summary_hook],
                    steps     = EVAL_STEPS)
                
                EPOCH_DISPLAY = int( int(results_val['global_step']) / 153.25 )
                LR_DISPLAY    = LR_INITIAL * np.power(LRDECAY_RATE, int(int(results_val['global_step'])/LRDECAY_STEPS))
                
                print('Epoch = {}; Step = {} / learning_rate = {:.5f}; val loss = {:.5f};'.format(
                     EPOCH_DISPLAY, 
                     results_val['global_step'], 
                     LR_DISPLAY, 
                     results_val['loss']))
                
                dim                        = args.model_path + 'Step{}valloss{:.5f}'.format(results_val['global_step'], results_val['loss'])
                export_dir                 = nn.export_savedmodel(
                export_dir_base            = dim,
                serving_input_receiver_fn  = reader.serving_input_receiver_fn(reader_patch_shapes))
                print('Model saved to {}.'.format(export_dir))
                count_steps.append(results_val['global_step'])
                count_loss.append(results_val['loss'])

    except KeyboardInterrupt:
        pass

    # Arbitrary Input Shape during Export
    export_dir = nn.export_savedmodel(
        export_dir_base           = args.model_path,
        serving_input_receiver_fn = reader.serving_input_receiver_fn(
            {'features': {'x': [None, None, None, NUM_CHANNELS]},
             'labels':   {'y': [1]}}))
    print('Model saved to {}.'.format(export_dir))

    Step_Loss      = pd.DataFrame(list(zip(count_steps,count_loss)),
    columns        = ['Steps','val_loss'])
    Step_Loss.to_csv("ValidationLoss.csv", encoding='utf-8', index=False)



if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Argument Parser Setup
    parser = argparse.ArgumentParser(description='Binary Lungs Disease Classification')
    parser.add_argument('--run_validation',     default=True)
    parser.add_argument('--restart',            default=False, action='store_true')
    parser.add_argument('--verbose',            default=False, action='store_true')
   
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path',   '-p', default='/Local/scripts/lungs/classification/weights/01082019/Fold_1')
    parser.add_argument('--train_csv',    '-t', default='/Local/scripts/lungs/classification/feed/csvIIIh/Lung_CV-Training-Fold-1.csv')
    parser.add_argument('--val_csv',      '-v', default='/Local/scripts/lungs/classification/feed/csvIIIh/Lung_CV-Validation-Fold-1.csv')
    parser.add_argument('--config',             default='config.json')

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

    # Handle Restarting/Resuming Training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Train
    train(args)

    session.close()
