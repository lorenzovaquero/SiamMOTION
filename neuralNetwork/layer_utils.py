"""SiameseBranch_AlexNet.py: AlexNet-based network for extracting image features"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import logging

logger = logging.getLogger(__name__)

slim = tf.contrib.slim

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


MOVING_AVERAGE_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'sf_update_ops'


def convolutional_layer(input_tensor, num_filters, kernel_size, stride=1, dilation_rate=1, num_groups=1, use_bias=True,
                        weight_decay=0.0, biases_decay=0.0, init_method='xavier', name="conv", padding='VALID'):
    """Builds a convolutional layer that computes 'filters' features
    using a 'kernel_size'filter and a stride of 'stride'.
    Padding is NOT added to preserve width and height.
    It also includes batch normalization if it's not the last 'is_last' layer"""
    logger.debug('Building convolutional layer')

    assert stride == 1 or dilation_rate == 1, "stride must be 1 if dilation_rate > 1"

    if init_method.lower() == 'kaiming'.lower():
        initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

    elif init_method.lower() == 'xavier'.lower():
        initializer = slim.xavier_initializer()

    elif init_method.lower() == 'yangqing'.lower():
        initializer = slim.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)

    else:
        logger.error('Init method type ' + init_method + 'isn\'t supported!')
        initializer = slim.xavier_initializer()

    # TODO: Que el inicializador Kaiming (y Xavier) tenga en cuenta si se usa leaky ReLu (tal vez cambiando `factor`)?

    if weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    if biases_decay > 0:
        biases_regularizer = slim.l2_regularizer(biases_decay)
    else:
        biases_regularizer = None

    if use_bias:
        biases_initializer = tf.zeros_initializer()
    else:
        biases_initializer = None

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=weights_regularizer,
                        weights_initializer=initializer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=biases_initializer,
                        padding=padding,
                        trainable=True,
                        activation_fn=None,
                        normalizer_fn=None):

        if num_groups == 1:
            layer = slim.conv2d(input_tensor, num_filters, kernel_size, stride, rate=dilation_rate, scope=name)
        else:
            with tf.compat.v1.variable_scope(name):
                input_groups = tf.split(axis=3, num_or_size_splits=num_groups, value=input_tensor)
                conv_groups = [slim.conv2d(group, num_filters, kernel_size, stride, rate=dilation_rate,
                                           scope='group%d' % (i + 1)) for i, group in enumerate(input_groups)]
                layer = tf.concat(axis=3, values=conv_groups)

        return layer


def batch_normalization_layer(input_tensor, is_training, momentum=0.9, epsilon=1e-5, use_scale=True, reuse=None,
                              name='batch_normalization'):
    """Batch Normalization Layer"""

    logger.debug('Building batch normalization layer')

    # layer = tf.layers.batch_normalization(input_tensor, training=is_training, momentum=momentum,
    #                                       epsilon=epsilon, scale=use_scale, reuse=reuse, name=name)

    # NO consigo emular el BN de PyTorch, pero este se le parece y aparentemente funciona bien
    bn = tf.compat.v1.layers.BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon,
                                                scale=use_scale, name=name, trainable=is_training)
    layer = bn(input_tensor, training=is_training)

    return layer



def leaky_relu_layer(input_tensor, alpha=0.2, name='leaky_relu'):
    """Leaky ReLU Layer"""

    logger.debug('Building Leaky ReLU layer')

    layer = tf.nn.leaky_relu(input_tensor, alpha=alpha, name=name)

    return layer



def relu_layer(input_tensor, name='relu'):
    """Leaky ReLU Layer"""

    logger.debug('Building ReLU layer')

    layer = tf.nn.relu(input_tensor, name=name)

    return layer



def max_pooling_layer(input_tensor, kernel_size, stride, name='max_pooling', padding='VALID'):
    """Builds a max pooling layer with a 'kernel_size' filter and stride of 'stride'"""
    logger.debug('Building max pooling layer')

    layer = slim.max_pool2d(input_tensor, kernel_size, stride, padding=padding, scope=name)

    return layer

