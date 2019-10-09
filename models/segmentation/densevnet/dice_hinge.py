# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer


def dice(prediction, ground_truth, weight_map=None):
    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    # if weight_map is not None:
    #    weight_map_nclasses = tf.reshape(
    #        tf.tile(weight_map, [n_classes]), prediction.get_shape())
    #    dice_numerator = 2.0 * tf.sparse_reduce_sum(
    #        weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
    # else:
    dice_numerator = 2.0 * tf.sparse_reduce_sum(
        one_hot * prediction, reduction_axes=[0])
    dice_denominator = \
        tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
        tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    # dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)
    [tf.summary.scalar('Dice{}'.format(i),d) for i,d in enumerate(tf.unstack(dice_score,0))]
    dice_score=tf.Print(dice_score,[dice_score],summarize=10,message='dice')
    h1=tf.square(tf.minimum(0.1,dice_score)*10-1)
    h2=tf.square(tf.minimum(0.01,dice_score)*100-1)


    return 1.0 - tf.reduce_mean(dice_score) + \
           tf.reduce_mean(h1)*10 + \
           tf.reduce_mean(h2)*10



