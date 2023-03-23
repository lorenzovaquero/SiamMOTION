from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging

from .layer_utils import convolutional_layer, batch_normalization_layer
from inference.inference_utils_tf import crop_tensor_center

logger = logging.getLogger(__name__)

slim = tf.contrib.slim

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class NeckLayer(object):
    """Creates a Neck Layer. Its purpose is emulate PySOT's one: take several features tensors to reduce their channels
    to the same number (e.g. 256)"""


    def __init__(self, parameters):
        super(NeckLayer, self).__init__()
        self.parameters = parameters

    def _build_neck(self, input_tensor, num_channels, is_training, crop_center=None, op_name='downsample'):
        x = convolutional_layer(input_tensor, num_filters=num_channels, kernel_size=1, stride=1,
                                padding='VALID', use_bias=False,
                                weight_decay=self.parameters.convolutionWeightDecay,
                                biases_decay=self.parameters.convolutionBiasDecay,
                                init_method=self.parameters.initMethod, name=op_name + '.conv')
        x = batch_normalization_layer(x, is_training=is_training,
                                      momentum=self.parameters.batchNormalizationWeightDecay,
                                      epsilon=self.parameters.epsilon,
                                      use_scale=True, name=op_name + '.bn')

        if crop_center is not None: 
                x = crop_tensor_center(input_tensor, crop_size=crop_center)

        return x



    def build(self, input_tensor, is_training, crop_center=None, num_output_channels=256, name='neck'):
        """Creates a Neck Layer. Its purpose is emulate PySOT's one: take several features tensors to reduce their channels
        to the same number (e.g. 256)"""

        logger.debug('Building Neck layer \'' + name + '\'')

        self.is_training = is_training
        self.input_tensor = input_tensor
        self.crop_center = crop_center

        if type(input_tensor) in [list, tuple]:
            multiple_inputs = True
        else:
            multiple_inputs = False

        with tf.compat.v1.variable_scope(name):
            if multiple_inputs:
                self.output_tensor = []
                for i, features_tensor in enumerate(self.input_tensor):
                    self.output_tensor.append(self._build_neck(features_tensor, num_channels=num_output_channels,
                                                               is_training=self.is_training,
                                                               crop_center=self.crop_center, op_name='downsample' + str(i)))

            else:
                self.output_tensor = self._build_neck(self.input_tensor, num_channels=num_output_channels, is_training=self.is_training,
                                                      crop_center=self.crop_center, op_name='downsample')

        return self.output_tensor