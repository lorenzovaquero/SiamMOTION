"""RPNLayer.py: Generates classification scores and regression coordinates comparing two feature maps"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from neuralNetwork.layer_utils import convolutional_layer
from neuralNetwork.SimilarityLayer import SimilarityLayer
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


class RPNLayer(SimilarityLayer):
    """Generates classification scores and regression coordinates comparing two feature maps"""

    variable_list = ['RPN/classification/conv_exemplar/weights:0', 'RPN/classification/conv_exemplar/biases:0',
                     'RPN/classification/conv_searcharea/weights:0', 'RPN/classification/conv_searcharea/biases:0',
                     'RPN/regression/conv_exemplar/weights:0', 'RPN/regression/conv_exemplar/biases:0',
                     'RPN/regression/conv_searcharea/weights:0', 'RPN/regression/conv_searcharea/biases:0',
                     'RPN/regression/adjust/weights:0', 'RPN/regression/adjust/biases:0']
    name = 'RPN'

    def __init__(self, parameters):
        super(RPNLayer, self).__init__(parameters)

    @classmethod
    def _get_scope_name_and_N_values(cls, is_regression):
        if is_regression:
            N = 4
            scope_name = "regression"
        else:
            N = 2
            scope_name = "classification"

        return scope_name, N

    def __build_RPN_branch(self, exemplar_tensor, searcharea_tensor, is_regression, is_training=False):
        """Builds the similarity operation for one of the RPN branches (cls or reg).
            It receives two feature tensors, converts them to 'kernels' and outputs the similarityy."""

        branch_exemplar_kernel = self._build_exemplar_kernel(exemplar_tensor=exemplar_tensor,
                                                             is_regression=is_regression, is_training=False)
        branch_searcharea_kernel = self._build_searcharea_kernel(searcharea_tensor=searcharea_tensor,
                                                                 is_regression=is_regression, is_training=is_training)

        branch_out = self._build_cross_correlation(exemplar_tensor_kernel=branch_exemplar_kernel,
                                                   searcharea_tensor_kernel=branch_searcharea_kernel,
                                                   is_regression=is_regression, is_training=is_training)

        branch_out = self._build_output_adjust(xcorr_output=branch_out, is_regression=is_regression, is_training=is_training)

        return branch_out

    def _build_exemplar_kernel(self, exemplar_tensor, is_regression, is_training=False):
        """Receives a exemplar feature tensor and transforms it into a (cls or reg) 'kernel' for
            a RPN branch, in order to use it in the "__build_cross_correlation" function."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        with tf.compat.v1.variable_scope(scope_name):
            input_num_channels = int(exemplar_tensor.shape[-1])
            branch_exemplar_kernel = convolutional_layer(input_tensor=exemplar_tensor, num_filters=input_num_channels * N * self.parameters.numAnchors,
                                                    kernel_size=[3, 3], stride=1, num_groups=1, use_bias=True,
                                                    weight_decay=self.parameters.convolutionWeightDecay,
                                                    init_method=self.parameters.initMethod, name='conv_exemplar')

            # exemplar [?, 4, 4, 5120], searcharea [?, 20, 20, 256]

            k = int(branch_exemplar_kernel.shape[-1])
            w = int(branch_exemplar_kernel.shape[-2])
            h = int(branch_exemplar_kernel.shape[-3])

            #  exemplar [?, 4, 4, Nk, 256]
            branch_exemplar_kernel = tf.reshape(branch_exemplar_kernel, [-1, h, w, N * self.parameters.numAnchors, input_num_channels])
            #  exemplar [?, 4, 4, 256, Nk]
            branch_exemplar_kernel = tf.transpose(branch_exemplar_kernel, [0, 1, 2, 4, 3])

            branch_exemplar_kernel = self._cross_correlation_depthwise_kernel(exemplar_features=branch_exemplar_kernel)
            # Kernel now has shape [batch_size, kernel_height, kernel_width, kernel_channels, out_channels]

        return branch_exemplar_kernel

    def _build_searcharea_kernel(self, searcharea_tensor, is_regression, is_training=False):
        """Receives a searcharea feature tensor and transforms it into a (cls or reg) 'kernel' for
            a RPN branch, in order to use it in the "__build_cross_correlation" function."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        with tf.compat.v1.variable_scope(scope_name):
            input_num_channels = int(searcharea_tensor.shape[-1])

            branch_searcharea_kernel = convolutional_layer(input_tensor=searcharea_tensor, num_filters=input_num_channels,
                                                      kernel_size=[3, 3], stride=1, num_groups=1, use_bias=True,
                                                      weight_decay=self.parameters.convolutionWeightDecay,
                                                      init_method=self.parameters.initMethod, name='conv_searcharea')

        return branch_searcharea_kernel

    def _build_cross_correlation(self, exemplar_tensor_kernel, searcharea_tensor_kernel, is_regression, is_training=False):
        """Receives a exemplar features 'kernel' and a searcharea features 'kernel' and and outputs
            the similarity score map of one of the RPN branches (cls or reg)."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        with tf.compat.v1.variable_scope(scope_name):
            logger.debug('Exemplar que entra a conv {}: "{}"'.format(scope_name, str(exemplar_tensor_kernel)))
            logger.debug('Searcharea que entra a conv {}: "{}"'.format(scope_name, str(searcharea_tensor_kernel)))
            out_reg = self._cross_correlation_layer_depthwise(exemplar_features=exemplar_tensor_kernel,
                                                              searcharea_features=searcharea_tensor_kernel,
                                                              name='out')

            logger.debug('Score que sale de {}: "{}"'.format(scope_name, str(out_reg)))
            #  score [?, 17, 17, Nk]

        return out_reg

    def _build_output_adjust(self, xcorr_output, is_regression, is_training=False):
        """Receives the ouptut of a RPN branch and adjusts it."""
        scope_name, N = self._get_scope_name_and_N_values(is_regression)

        with tf.compat.v1.variable_scope(scope_name):
            if is_regression:
                out_reg = convolutional_layer(input_tensor=xcorr_output, num_filters=int(xcorr_output.shape[-1]),
                                              kernel_size=[1, 1], stride=1, num_groups=1, use_bias=True,
                                              weight_decay=self.parameters.convolutionWeightDecay,
                                              init_method=self.parameters.initMethod, name='adjust')
            else:
                out_reg = xcorr_output

            return out_reg

    def build(self, exemplar_tensor, searcharea_tensor, is_training=False, scope_name=name):
        with tf.compat.v1.variable_scope(scope_name):
            cls_out = self.__build_RPN_branch(exemplar_tensor=exemplar_tensor,
                                              searcharea_tensor=searcharea_tensor,
                                              is_regression=False,
                                              is_training=is_training)
            reg_out = self.__build_RPN_branch(exemplar_tensor=exemplar_tensor,
                                              searcharea_tensor=searcharea_tensor,
                                              is_regression=True,
                                              is_training=is_training)

        return cls_out, reg_out

    def build_kernel(self, exemplar_tensor, is_training=False, scope_name=name):
        """Receives a exemplar feature tensor and transforms it into RPN cls and reg 'kernels'
            in order to use them in the "build_from_kernel" function."""
        with tf.compat.v1.variable_scope(scope_name):
            cls_exemplar_kernel = self._build_exemplar_kernel(exemplar_tensor=exemplar_tensor,
                                                              is_regression=False, is_training=is_training)

            reg_exemplar_kernel = self._build_exemplar_kernel(exemplar_tensor=exemplar_tensor,
                                                              is_regression=True, is_training=is_training)

        return cls_exemplar_kernel, reg_exemplar_kernel

    def build_from_kernel(self, exemplar_kernel, searcharea_tensor, is_training=False, scope_name=name):
        """Receives a exemplar features 'kernel' (cls and reg) and a searcharea features tensor and and outputs
            a similarity score map."""
        if not isinstance(exemplar_kernel, (list, tuple)):
            raise ValueError('The "exemplar_kernel" argument must be a list containing the cls and reg RPN exemplar '
                             'kernels. The provided object is: {}'.format(str(exemplar_kernel)))

        if len(exemplar_kernel) != 2:
            raise ValueError('The "exemplar_kernel" argument must contain 2 tensors (cls_exemplar_kernel and '
                             'reg_exemplar_kernel). The provided object contains {}: {}'.format(len(exemplar_kernel), str(exemplar_kernel)))

        cls_exemplar_kernel = exemplar_kernel[0]
        reg_exemplar_kernel = exemplar_kernel[1]

        with tf.compat.v1.variable_scope(scope_name):
            # Classification
            conv_cls_searcharea = self._build_searcharea_kernel(searcharea_tensor=searcharea_tensor,
                                                                is_regression=False, is_training=is_training)
            cls_out = self._build_cross_correlation(exemplar_tensor_kernel=cls_exemplar_kernel,
                                                    searcharea_tensor_kernel=conv_cls_searcharea,
                                                    is_regression=False, is_training=is_training)
            cls_out = self._build_output_adjust(xcorr_output=cls_out, is_regression=False, is_training=is_training)

            # Regression
            conv_reg_searcharea = self._build_searcharea_kernel(searcharea_tensor=searcharea_tensor,
                                                                is_regression=True, is_training=is_training)
            reg_out = self._build_cross_correlation(exemplar_tensor_kernel=reg_exemplar_kernel,
                                                    searcharea_tensor_kernel=conv_reg_searcharea,
                                                    is_regression=True, is_training=is_training)
            reg_out = self._build_output_adjust(xcorr_output=reg_out, is_regression=True, is_training=is_training)

        return cls_out, reg_out