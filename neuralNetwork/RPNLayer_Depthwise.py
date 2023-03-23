"""RPNLayer_Depthwise.py: It is like RPNLayer.py but performing a DepthwiseXCorr between exemplar and searcharea
(instead of a XCorr) just like PySOT does"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from neuralNetwork.RPNLayer import RPNLayer
from neuralNetwork.layer_utils import convolutional_layer, batch_normalization_layer, relu_layer
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


class RPNLayer_Depthwise(RPNLayer):
    """Generates classification scores and regression coordinates comparing two feature maps using a DepthwiseXCorr"""

    variable_list = ['kernels_creation/RPN/classification/conv_exemplar/weights', 'kernels_creation/RPN/classification/batch_normalization_exemplar/gamma',
                     'kernels_creation/RPN/classification/batch_normalization_exemplar/beta', 'kernels_creation/RPN/classification/batch_normalization_exemplar/moving_mean',
                     'kernels_creation/RPN/classification/batch_normalization_exemplar/moving_variance', 'kernels_creation/RPN/regression/conv_exemplar/weights',
                     'kernels_creation/RPN/regression/batch_normalization_exemplar/gamma', 'kernels_creation/RPN/regression/batch_normalization_exemplar/beta',
                     'kernels_creation/RPN/regression/batch_normalization_exemplar/moving_mean', 'kernels_creation/RPN/regression/batch_normalization_exemplar/moving_variance',
                     'xcorr/RPN/classification/conv_searcharea/weights', 'xcorr/RPN/classification/batch_normalization_searcharea/gamma', 'xcorr/RPN/classification/batch_normalization_searcharea/beta',
                     'xcorr/RPN/classification/batch_normalization_searcharea/moving_mean', 'xcorr/RPN/classification/batch_normalization_searcharea/moving_variance',
                     'xcorr/RPN/regression/conv_searcharea/weights', 'xcorr/RPN/regression/batch_normalization_searcharea/gamma',
                     'xcorr/RPN/regression/batch_normalization_searcharea/beta', 'xcorr/RPN/regression/batch_normalization_searcharea/moving_mean',
                     'xcorr/RPN/regression/batch_normalization_searcharea/moving_variance', 'xcorr/RPN/classification/adjust/conv1/weights',
                     'xcorr/RPN/classification/adjust/batch_normalization/gamma', 'xcorr/RPN/classification/adjust/batch_normalization/beta',
                     'xcorr/RPN/classification/adjust/batch_normalization/moving_mean', 'xcorr/RPN/classification/adjust/batch_normalization/moving_variance',
                     'xcorr/RPN/classification/adjust/conv2/weights', 'xcorr/RPN/classification/adjust/conv2/biases', 'xcorr/RPN/regression/adjust/conv1/weights',
                     'xcorr/RPN/regression/adjust/batch_normalization/gamma', 'xcorr/RPN/regression/adjust/batch_normalization/beta',
                     'xcorr/RPN/regression/adjust/batch_normalization/moving_mean', 'xcorr/RPN/regression/adjust/batch_normalization/moving_variance',
                     'xcorr/RPN/regression/adjust/conv2/weights', 'xcorr/RPN/regression/adjust/conv2/biases']
    name = 'RPN'

    def __init__(self, parameters):
        super(RPNLayer_Depthwise, self).__init__(parameters)
        self.num_channels = None


    def _build_exemplar_kernel(self, exemplar_tensor, is_regression, is_training=False):
        """Receives a exemplar feature tensor and transforms it into a (cls or reg) 'kernel' for
            a RPN branch, in order to use it in the "__build_cross_correlation" function."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        with tf.compat.v1.variable_scope(scope_name):
            input_num_channels = int(exemplar_tensor.shape[-1])

            branch_exemplar_kernel = convolutional_layer(input_tensor=exemplar_tensor, num_filters=input_num_channels,
                                                    kernel_size=[3, 3], stride=1, num_groups=1, use_bias=False,
                                                    weight_decay=self.parameters.convolutionWeightDecay,
                                                    init_method=self.parameters.initMethod, name='conv_exemplar')
            branch_exemplar_kernel = batch_normalization_layer(branch_exemplar_kernel, is_training=is_training,
                                                               momentum=self.parameters.batchNormalizationWeightDecay,
                                                               epsilon=self.parameters.epsilon, use_scale=True,
                                                               name='batch_normalization_exemplar')
            branch_exemplar_kernel = relu_layer(branch_exemplar_kernel, name='relu')


            #  exemplar [?, 4, 4, 256], searcharea [?, 20, 20, 256]

            k = int(branch_exemplar_kernel.shape[-1])
            w = int(branch_exemplar_kernel.shape[-2])
            h = int(branch_exemplar_kernel.shape[-3])

            #  exemplar [?, 256, 4, 4]
            branch_exemplar_kernel = tf.transpose(branch_exemplar_kernel, [0, 3, 1, 2])

            #  exemplar [?*256, 4, 4, 1, 1]
            branch_exemplar_kernel = tf.reshape(branch_exemplar_kernel, [-1, h, w, 1, 1])

            branch_exemplar_kernel = self._cross_correlation_depthwise_kernel(exemplar_features=branch_exemplar_kernel)
            # Kernel now has shape [batch_size, kernel_height, kernel_width, kernel_channels, out_channels]

        return branch_exemplar_kernel

    def _build_searcharea_kernel(self, searcharea_tensor, is_regression, is_training=False):
        """Receives a searcharea feature tensor and transforms it into a (cls or reg) 'kernel' for
            a RPN branch, in order to use it in the "__build_cross_correlation" function."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        with tf.compat.v1.variable_scope(scope_name):
            input_num_channels = int(searcharea_tensor.shape[-1])
            self.num_channels = input_num_channels

            branch_searcharea_kernel = convolutional_layer(input_tensor=searcharea_tensor, num_filters=input_num_channels,
                                                      kernel_size=[3, 3], stride=1, num_groups=1, use_bias=False,
                                                      weight_decay=self.parameters.convolutionWeightDecay,
                                                      init_method=self.parameters.initMethod, name='conv_searcharea')
            branch_searcharea_kernel = batch_normalization_layer(branch_searcharea_kernel, is_training=is_training,
                                                               momentum=self.parameters.batchNormalizationWeightDecay,
                                                               epsilon=self.parameters.epsilon, use_scale=True,
                                                               name='batch_normalization_searcharea')
            branch_searcharea_kernel = relu_layer(branch_searcharea_kernel, name='relu')

            w = int(branch_searcharea_kernel.shape[-2])
            h = int(branch_searcharea_kernel.shape[-3])

            # searcharea [?*256, 22, 22, 1]
            branch_searcharea_kernel = tf.transpose(branch_searcharea_kernel, [0, 3, 1, 2])   #  searcharea [?, 256, 22, 22]
            branch_searcharea_kernel = tf.reshape(branch_searcharea_kernel, [-1, h, w, 1])  # searcharea [?*256, 22, 22, 1]

        return branch_searcharea_kernel

    def _build_cross_correlation(self, exemplar_tensor_kernel, searcharea_tensor_kernel, is_regression, is_training=False):
        """Receives a exemplar features 'kernel' and a searcharea features 'kernel' and and outputs
            the similarity score map of one of the RPN branches (cls or reg)."""
        out = super(RPNLayer_Depthwise, self)._build_cross_correlation(exemplar_tensor_kernel=exemplar_tensor_kernel,
                                                                       searcharea_tensor_kernel=searcharea_tensor_kernel,
                                                                       is_regression=is_regression,
                                                                       is_training=is_training)
        #  score [?*256, 17, 17, 1]

        num_channels = self.num_channels
        w = int(out.shape[-2])
        h = int(out.shape[-3])

        assert int(out.shape[-1]) == 1

        #  score [?, 17, 17, 256]
        out = tf.reshape(out[..., 0], [-1, num_channels, h, w])  # score [?, 256, 17, 17]
        out = tf.transpose(out, [0, 2, 3, 1])  # score [?, 17, 17, 256]

        return out

    def _build_output_adjust(self, xcorr_output, is_regression, is_training=False):
        """Receives the ouptut of a RPN branch and adjusts it."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        num_filters = int(xcorr_output.shape[-1])

        with tf.compat.v1.variable_scope(scope_name):
            with tf.compat.v1.variable_scope('adjust'):

                out = convolutional_layer(input_tensor=xcorr_output, num_filters=num_filters,
                                          kernel_size=[1, 1], stride=1, num_groups=1, use_bias=False,
                                          weight_decay=self.parameters.convolutionWeightDecay,
                                          init_method=self.parameters.initMethod, name='conv1')
                out = batch_normalization_layer(out, is_training=is_training,
                                                momentum=self.parameters.batchNormalizationWeightDecay,
                                                epsilon=self.parameters.epsilon, use_scale=True,
                                                name='batch_normalization')
                out = relu_layer(out, name='relu')
                out = convolutional_layer(input_tensor=out, num_filters=N * self.parameters.numAnchors,
                                          kernel_size=[1, 1], stride=1, num_groups=1, use_bias=True,
                                          weight_decay=self.parameters.convolutionWeightDecay,
                                          init_method=self.parameters.initMethod, name='conv2')

            return out