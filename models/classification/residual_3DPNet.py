from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from dltk.core.residual_unit import vanilla_residual_unit_3d
from dltk.core.upsample import linear_upsample_3d
from dltk.core.activations import leaky_relu



# Auxiliary Functions
def upsample_and_concat(inputs, inputs2, strides=(2, 2, 2)):
    """Upsampling and concatenation layer according to [1].

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.

    Args:
        inputs (TYPE): Input features to be upsampled.
        inputs2 (TYPE): Higher resolution features from the encoder to
            concatenate.
        strides (tuple, optional): Upsampling factor for a strided transpose
            convolution.

    Returns:
        tf.Tensor: Upsampled feature tensor
    """
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), \
        'Ranks of input and input2 differ'

    # Upsample inputs
    inputs = linear_upsample_3d(inputs, strides)

    return tf.concat(axis=-1, values=[inputs2, inputs])


# Target Model
def residual_3DPNet (inputs,
                     num_classes,
                     mode                        = tf.estimator.ModeKeys.EVAL,
                     
                     seg__num_res_units          = 1,
                     seg__filters                = (16, 32, 64, 128),
                     seg__strides                = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                     seg__use_bias               = False,
                     seg__activation             = leaky_relu,
                     seg__kernel_initializer     = tf.initializers.variance_scaling(distribution='uniform'),
                     seg__bias_initializer       = tf.zeros_initializer(),
                     seg__kernel_regularizer     = None,
                     seg__bias_regularizer       = None,
                     seg__bottleneck             = False,
                     
                     clf__num_res_units          = 1,
                     clf__filters                = (16, 32, 64, 128),
                     clf__strides                = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)), 
                     clf__use_bias               = False,
                     clf__activation             = tf.nn.relu6,
                     clf__kernel_initializer     = tf.initializers.variance_scaling(distribution='uniform'),
                     clf__bias_initializer       = tf.zeros_initializer(),
                     clf__kernel_regularizer     = None, 
                     clf__bias_regularizer       = None):

    
    # RESIDUAL 3D U-NET (residual_3D_unet) // SEGMENTATION {
    """
    Image segmentation network based on a flexible UNET architecture [1]
    using residual units [2] as feature extractors. Downsampling and
    upsampling of features is done via strided convolutions and transpose
    convolutions, respectively. On each resolution scale s are
    num_residual_units with filter size = filters[s]. strides[s] determine
    the downsampling factor at each resolution scale.
     
    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.
    [2] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.
    
    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output classes.
        num_res_units (int, optional): Number of residual units at each
            resolution scale.
        filters (tuple, optional): Number of filters for all residual units at
            each resolution scale.
        strides (tuple, optional): Stride of the first unit on a resolution
            scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        activation (optional): A function to use as activation function.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.
    
    Returns:
        dict: dictionary of output tensors
    
    }"""
    # Input:  CT Patch Features 
    # Output: Segmentation Feature Maps
    
    
    # INITIALIZATION
    assert len(seg__strides) == len(seg__filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'Inputs are required to have a rank of 5.'
    
    seg__conv_params = {'padding':              'same',
                        'use_bias':             seg__use_bias,
                        'kernel_initializer':   seg__kernel_initializer,
                        'bias_initializer':     seg__bias_initializer,
                        'kernel_regularizer':   seg__kernel_regularizer,
                        'bias_regularizer':     seg__bias_regularizer}
    x = inputs
    
    # Initial Convolution with (seg__filters[0])
    x = tf.layers.conv3d(inputs       = x,
                         filters      = seg__filters[0],
                         kernel_size  = (3, 3, 3),
                         strides      = seg__strides[0],
                         **seg__conv_params)
    tf.logging.info('Segmentation Network: Initial Conv3D Tensor Shape: {}'.format(x.get_shape()))
    
    
    
    
    # FEATURE EXTRACTOR // ENCODER:
    # Residual Blocks with (seg__num_res_units) at Different Resolution Scales (res_scales)
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(seg__filters)):
     
        # Features are Downsampled via Strided Convolutions ('seg__strides')
        with tf.variable_scope('enc_unit_{}_0'.format(res_scale)):
            x = vanilla_residual_unit_3d(
                inputs            = x,
                out_filters       = seg__filters[res_scale],
                strides           = seg__strides[res_scale],
                activation        = seg__activation,
                mode              = mode)
        saved_strides.append(seg__strides[res_scale])
        
        for i in range(1, seg__num_res_units):
            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3d(
                    inputs        = x,
                    out_filters   = seg__filters[res_scale],
                    strides       = (1, 1, 1),
                    activation    = seg__activation,
                    mode          = mode)
        res_scales.append(x)
        tf.logging.info('Segmentation Network: Encoder at "res_scale" {} Tensor Shape: {}'.format(
            res_scale, x.get_shape()))
    
    
    
    
    # RESTORE SPATIAL DIMENSION // DECODER:
    # Upsample and Concatenate Layers and Reconstruct Predictions to Higher Resolution Scales
    for res_scale in range(len(seg__filters) - 2, -1, -1):
        
        with tf.variable_scope('up_concat_{}'.format(res_scale)):
            x = upsample_and_concat(
                inputs           = x,
                inputs2          = res_scales[res_scale],
                strides          = saved_strides[res_scale])
        
        for i in range(0, seg__num_res_units):
            with tf.variable_scope('dec_unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3d(
                    inputs       = x,
                    out_filters  = seg__filters[res_scale],
                    strides      = (1, 1, 1),
                    mode         = mode)
        tf.logging.info('Segmentation Network: Decoder at "res_scale" {} Tensor Shape: {}'.format(
            res_scale, x.get_shape()))
    
    
    
    
    # BOTTLENECK CONVOLUTION
    if seg__bottleneck:
     with tf.variable_scope('last'):
        
         x = tf.layers.conv3d(inputs         = x,
                              filters        = num_classes,
                              kernel_size    = (1, 1, 1),
                              strides        = (1, 1, 1),
                              **seg__conv_params)
     tf.logging.info('Segmentation Network: Bottleneck Output Tensor Shape: {}'.format(x.get_shape()))
    
    
    
    
    # OUTPUT
    residual3Dunet_output = x
    
    # VISUALIZE SEGMENTATION FEATURE MAPS
    F           = residual3Dunet_output
    shape       = F.get_shape().as_list()
    ydim        = shape[2]
    xdim        = shape[3]
    featuremaps = shape[4]
    F = tf.slice(F,(0,0,0,0,0),(1,1,-1,-1,-1)) 
    F = tf.reshape(F,(ydim,xdim,featuremaps))
    ydim += 2
    xdim += 2
    F = tf.image.resize_image_with_crop_or_pad(F,ydim,xdim)
    F = tf.reshape(F,(ydim,xdim,2,4)) 
    F = tf.transpose(F,(2,0,3,1))
    F = tf.reshape(F,(1,2*ydim,4*xdim,1))
    tf.summary.image('Segmentation Feature Maps', F, 20)
    
    
    
    
    
    
    
    
    
    
    # 3D RESNET (resnet_3d) // CLASSIFICATION {
    """
    Regression/classification network based on a flexible resnet
    architecture [1] using residual units proposed in [2]. The downsampling
    of features is done via strided convolutions. On each resolution scale s
    are num_convolutions with filter size = filters[s]. strides[s]
    determine the downsampling factor at each resolution scale.
    
    [1] K. He et al. Deep residual learning for image recognition. CVPR 2016.
    [2] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.
    
    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output channels or classes.
        num_res_units (int, optional): Number of residual units per resolution
            scale.
        filters (tuple, optional): Number of filters for all residual units at
            each resolution scale.
        strides (tuple, optional): Stride of the first unit on a resolution
            scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        activation (optional): A function to use as activation function.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.
    
    Returns:
        dict: dictionary of output tensors
    
    }"""
    # Input:  CT Patch Features (concat) Segmentation Feature Maps
    # Output: Classification Scores (logits, y_prob, y_)


    # INITIALIZATION
    resnet3D_output = {}
    assert len(clf__strides) == len(clf__filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'Inputs are required to have a rank of 5.'
    
    relu_op = tf.nn.relu6
    
    clf__conv_params = {'padding':              'same',
                        'use_bias':             clf__use_bias,
                        'kernel_initializer':   clf__kernel_initializer,
                        'bias_initializer':     clf__bias_initializer,
                        'kernel_regularizer':   clf__kernel_regularizer,
                        'bias_regularizer':     clf__bias_regularizer}
    






    # Concatenated CT + BatchNorm(Segmentation Features) --> Input Feed 
    residual3Dunet_output = tf.layers.batch_normalization(inputs=residual3Dunet_output, axis=4)
    x = tf.concat(values=[inputs, residual3Dunet_output], axis=4)   



    # Initial Convolution with (clf__filters[0])
    k = [s * 2 if s > 1 else 3 for s in clf__strides[0]]
    x = tf.layers.conv3d(x, clf__filters[0], k, clf__strides[0], **clf__conv_params)
    tf.logging.info('Classification Network: Initial Conv3D Tensor Shape: {}'.format(x.get_shape()))
    
    

    # VISUALIZE PRIMARY CLASSIFICATION FEATURE MAPS
    F1          = x
    shape       = F1.get_shape().as_list()
    ydim        = shape[2]
    xdim        = shape[3]
    featuremaps = shape[4]
    F1 = tf.slice(F1,(0,0,0,0,0),(1,1,-1,-1,-1)) 
    F1 = tf.reshape(F1,(ydim,xdim,featuremaps))
    ydim += 2
    xdim += 2
    F1 = tf.image.resize_image_with_crop_or_pad(F1,ydim,xdim)
    F1 = tf.reshape(F1,(ydim,xdim,2,4)) 
    F1 = tf.transpose(F1,(2,0,3,1))
    F1 = tf.reshape(F1,(1,2*ydim,4*xdim,1))
    tf.summary.image('Primary Classification Feature Maps', F1, 20)




       
    
    # FEATURE EXTRACTOR // ENCODER:
    # Residual Blocks with (clf__num_res_units) at Different Resolution Scales (res_scales)
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(clf__filters)):
        
        # Features are Downsampled via Strided Convolutions ('clf__strides')
        with tf.variable_scope('unit_{}_0'.format(res_scale)):
            x = vanilla_residual_unit_3d(
                inputs                = x,
                out_filters           = clf__filters[res_scale],
                strides               = clf__strides[res_scale],
                activation            = clf__activation,
                mode                  = mode)
        saved_strides.append(clf__strides[res_scale])
        
        for i in range(1, clf__num_res_units):
            with tf.variable_scope('unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3d(
                    inputs            = x,
                    out_filters       = clf__filters[res_scale],
                    strides           = (1, 1, 1),
                    activation        = clf__activation,
                    mode              = mode)
        res_scales.append(x)
        tf.logging.info('Classification Network: Encoder at "res_scale" {} Tensor Shape: {}'.format(
            res_scale, x.get_shape()))




    
    # GLOBAL POOLING + FINAL LAYER
    with tf.variable_scope('pool'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = relu_op(x)
        
        axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
        x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')
        
        tf.logging.info('Classification Network: Global Pooling Tensor Shape: {}'.format(x.get_shape()))
    
    with tf.variable_scope('last'):
        x = tf.layers.dense(inputs                = x,
                            units                 = num_classes,
                            activation            = None,
                            use_bias              = clf__conv_params['use_bias'],
                            kernel_initializer    = clf__conv_params['kernel_initializer'],
                            bias_initializer      = clf__conv_params['bias_initializer'],
                            kernel_regularizer    = clf__conv_params['kernel_regularizer'],
                            bias_regularizer      = clf__conv_params['bias_regularizer'],
                            name                  = 'hidden_units')
        
        tf.logging.info('Classification Network: Output Tensor Shape: {}'.format(x.get_shape()))



    
    # OUTPUT
    resnet3D_output['logits'] = x
    
    with tf.variable_scope('pred'):
        
        y_prob = tf.nn.softmax(x)
        resnet3D_output['y_prob'] = y_prob
        
        y_ = tf.argmax(x, axis=-1) \
            if num_classes > 1 \
            else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        resnet3D_output['y_'] = y_
    
    return resnet3D_output





'''
Model Summary:                                                                     < Tensor Shape >

INFO:tensorflow:Segmentation Network: Initial Conv3D Tensor Shape:               (1, 128, 128, 128, 16)
INFO:tensorflow:Segmentation Network: Encoder at "res_scale" 1 Tensor Shape:     (1, 64, 64, 64, 64)
INFO:tensorflow:Segmentation Network: Encoder at "res_scale" 2 Tensor Shape:     (1, 32, 32, 32, 128)
INFO:tensorflow:Segmentation Network: Encoder at "res_scale" 3 Tensor Shape:     (1, 16, 16, 16, 256)
INFO:tensorflow:Segmentation Network: Encoder at "res_scale" 4 Tensor Shape:     (1, 16, 16, 16, 512)
INFO:tensorflow:Upsampling from (1, 16, 16, 16, 512) to                          [1, 16, 16, 16, 512]
INFO:tensorflow:Segmentation Network: Decoder at "res_scale" 3 Tensor Shape:     (1, 16, 16, 16, 256)
INFO:tensorflow:Upsampling from (1, 16, 16, 16, 256) to                          [1, 32, 32, 32, 256]
INFO:tensorflow:Segmentation Network: Decoder at "res_scale" 2 Tensor Shape:     (1, 32, 32, 32, 128)
INFO:tensorflow:Upsampling from (1, 32, 32, 32, 128) to                          [1, 64, 64, 64, 128]
INFO:tensorflow:Segmentation Network: Decoder at "res_scale" 1 Tensor Shape:     (1, 64, 64, 64, 64)
INFO:tensorflow:Upsampling from (1, 64, 64, 64, 64) to                           [1, 128, 128, 128, 64]
INFO:tensorflow:Segmentation Network: Decoder at "res_scale" 0 Tensor Shape:     (1, 128, 128, 128, 16)

INFO:tensorflow:Classification Network: Initial Conv3D Tensor Shape:             (1, 128, 128, 128, 16)
INFO:tensorflow:Classification Network: Encoder at "res_scale" 1 Tensor Shape:   (1, 64, 64, 64, 32)
INFO:tensorflow:Classification Network: Encoder at "res_scale" 2 Tensor Shape:   (1, 32, 32, 32, 64)
INFO:tensorflow:Classification Network: Encoder at "res_scale" 3 Tensor Shape:   (1, 16, 16, 16, 128)
INFO:tensorflow:Classification Network: Encoder at "res_scale" 4 Tensor Shape:   (1, 8, 8, 8, 256)
INFO:tensorflow:Classification Network: Global Pooling Tensor Shape:             (1, 256)
INFO:tensorflow:Classification Network: Output Tensor Shape:                     (1, 2)
'''