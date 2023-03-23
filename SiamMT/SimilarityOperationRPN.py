from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

import inference.inference_utils_tf as inference_utils
from SiamMT.RPNFactory import RPNFactory
from inference.RPN_utils_tf import reshape_prediction_as_anchors, reshape_anchor_prediction_as_prediction

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class SimilarityOperationRPN(object):

    def __init__(self, parameters, is_training=False):
        self.parameters = parameters
        self.is_training = is_training

        self.RPN_layer = RPNFactory.get_rpn(self.parameters.rpnType)(parameters=self.parameters)

        self.exemplar_features_tensor = None
        self.searcharea_features_tensor = None

        self.num_targets = None

        self.exemplar_features_size = None
        self.searcharea_features_size = None

        self.searcharea_features_size_represented_inside_score = None

        self.cls_batch_tensor = None
        self.reg_batch_tensor = None

        self.scoremap = None

        self.scope = None


    def pre_build(self, exemplar_features_tensor, name='SimilarityOperation'):
        """Pre-builds the similarity operation. This is, it returns the exemplar features kernel that will
        be used in the furure. By doing this, we can cache the exemplar features kernel"""

        logger.debug('PRECreating SimilarityOperationRPN')

        with tf.compat.v1.variable_scope(name) as self.scope:
            # =========================== vvvvv FEATURES PREPROCESSING SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('features_preprocessing'):
                self.exemplar_features_tensor = exemplar_features_tensor  # num_targets, height, width, channels

                self.num_targets = tf.shape(self.exemplar_features_tensor)[0]

                self.exemplar_features_size = inference_utils.get_tensor_size(self.exemplar_features_tensor)
            # =========================== ^^^^^ FEATURES PREPROCESSING SECTION ^^^^^ ===========================

            # =========================== vvvvv RPN KERNELS CREATION SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('kernels_creation'):
                self.cls_exemplar_kernel, \
                self.reg_exemplar_kernel = self.RPN_layer.build_kernel(
                    exemplar_tensor=self.exemplar_features_tensor, is_training=self.is_training)
            # =========================== ^^^^^ RPN KERNELS CREATION SECTION ^^^^^ ===========================

        return self.cls_exemplar_kernel, self.reg_exemplar_kernel


    def build(self, cls_exemplar_kernel_tensor, reg_exemplar_kernel_tensor, searcharea_features_tensor):
        """Takes as input some searcharea features and the exemplar KERNEL computed on the self.pre_build()"""

        logger.debug('Creating SimilarityOperationRPN')

        self.cls_exemplar_kernel_placeholder = cls_exemplar_kernel_tensor
        self.reg_exemplar_kernel_placeholder = reg_exemplar_kernel_tensor
        self.searcharea_features_tensor = searcharea_features_tensor
        self.searcharea_features_size = inference_utils.get_tensor_size(self.searcharea_features_tensor)

        with tf.compat.v1.variable_scope(self.scope):
            # =========================== vvvvv XCORR SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('xcorr'):
                self.cls_batch_tensor, \
                self.reg_batch_tensor = self.RPN_layer.build_from_kernel(
                    exemplar_kernel=[self.cls_exemplar_kernel_placeholder, self.reg_exemplar_kernel_placeholder],
                    searcharea_tensor=self.searcharea_features_tensor,
                    is_training=self.is_training)

            # =========================== ^^^^^ XCORR SECTION ^^^^^ ===========================

            # =========================== vvvvv MISC. SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('misc'):
                self.response_size = inference_utils.get_tensor_size(self.cls_batch_tensor)

                pred_conf = reshape_prediction_as_anchors(pred_tensor=self.cls_batch_tensor, is_regression=False)
                score_pred = tf.nn.softmax(pred_conf, axis=-1)[:, :, 1]  # We only want the positive class
                self.scoremap = reshape_anchor_prediction_as_prediction(
                    pred_tensor=tf.expand_dims(score_pred, axis=-1), response_size=self.response_size)

                self.searcharea_features_size_represented_inside_score = inference_utils.get_tensor_represented_inside_xcorr(
                    tensor_size=self.searcharea_features_size,
                    filter_size=self.exemplar_features_size,
                    stride=1)
            # =========================== ^^^^^ MISC. SECTION ^^^^^ ===========================

        return self.cls_batch_tensor, self.reg_batch_tensor
