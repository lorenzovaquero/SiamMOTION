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

from inference.inference_utils_tf import preprocess_image, update_target_after_image_scaling, pad_frame_for_exact_valid_convolution, pad_frame_with_effective_size, pad_frame_for_exact_same_convolution

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class Preprocessor(object):

    def __init__(self, parameters):
        self.parameters = parameters

        self.frame_tensor = None
        self.target_tensor = None  # [num_targets, [center_vertical, center_horizontal, height, width]]
        self.backbone_filter_size = None
        self.backbone_stride = None
        self.frame_converted = None
        self.target_converted = None

        self.frame_scale_factor_tensor = None
        self.frame_pad_topleft_tensor = None

    def build(self, frame_tensor, target_tensor, backbone_filter_size, backbone_stride,
              reference_exemplar_size=(127, 127), reference_searcharea_size=(255, 255), name='Preprocessor'):

        self.frame_tensor = frame_tensor
        self.target_tensor = target_tensor  # [num_targets, [center_vertical, center_horizontal, height, width]]
        self.backbone_filter_size = backbone_filter_size
        self.backbone_stride = backbone_stride
        self.reference_exemplar_size = reference_exemplar_size
        self.reference_searcharea_size = reference_searcharea_size

        logger.debug('Creating Preprocessor')
        with tf.compat.v1.variable_scope(name):
            # Some ops don't work on GPU if tensor is UINT8
            casted_frame = tf.cast(self.frame_tensor, dtype=tf.float32)

            self.frame_converted, \
            self.frame_scale_factor_tensor = preprocess_image(casted_frame,
                                                            invert_rgb=self.parameters.invertRGBOrder,
                                                            set_size=self.parameters.setFrameSize,
                                                            minimum_size=self.parameters.minimumFrameSize,
                                                            center_on_0=self.parameters.zeroCenterPixelValues,
                                                            keep_pixel_values=self.parameters.keepPixelValues,
                                                            set_exact_size=self.parameters.setExactSize)

            self.target_converted = update_target_after_image_scaling(self.target_tensor, scale_factor=self.frame_scale_factor_tensor)

            self.frame_pad_topleft_tensor = tf.constant([0, 0])

            if len(self.frame_converted.shape) == 4:
                average_colors = tf.reduce_mean(self.frame_converted, axis=[1, 2], name='averageColors', keepdims=True)
            else:
                average_colors = tf.reduce_mean(self.frame_converted, axis=[0, 1], name='averageColors', keepdims=True)

            if self.parameters.padFrameEffectiveSize:
                self.frame_converted, pad_topleft = pad_frame_with_effective_size(self.frame_converted,
                                                                                  total_filter_size=self.backbone_filter_size,
                                                                                  total_stride=self.backbone_stride,
                                                                                  average_colors=average_colors,
                                                                                  pad_multiple_of_stride=self.parameters.adjustFrameValidConvolution)
                self.frame_pad_topleft_tensor += pad_topleft

            if self.parameters.adjustFrameValidConvolution and not self.parameters.padFrameEffectiveSize:
                self.frame_converted = pad_frame_for_exact_valid_convolution(self.frame_converted,
                                                                             total_filter_size=self.backbone_filter_size,
                                                                             total_stride=self.backbone_stride,
                                                                             average_colors=average_colors)

            if self.parameters.adjustFrameValidConvolution and self.parameters.padding == 'SAME':
                self.frame_converted, pad_topleft = pad_frame_for_exact_same_convolution(self.frame_converted,
                                                                                         total_filter_size=self.backbone_filter_size,
                                                                                         total_stride=self.backbone_stride,
                                                                                         average_colors=average_colors)
                self.frame_pad_topleft_tensor += pad_topleft

            self.target_converted = self.target_converted + tf.pad(tf.cast(self.frame_pad_topleft_tensor, dtype=tf.float32), paddings=[[0, 2]])


            if len(self.frame_converted.get_shape()) < 4:
                self.frame_converted = tf.expand_dims(self.frame_converted, axis=0)

        return self.frame_converted, self.target_converted
