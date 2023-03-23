from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

import inference.inference_utils_tf as inference_utils
from .BackboneFactory import BackboneFactory

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class FeatureExtractor(object):

    # https://pytorch.org/docs/stable/torchvision/models.html
    input_image_standardization_mean = np.array([0.485, 0.456, 0.406])  # RGB
    input_image_standardization_std = np.array([0.229, 0.224, 0.225])  # RGB

    def __init__(self, parameters, is_training=False):
        self.parameters = parameters
        self.is_training = is_training

        self.backbone = BackboneFactory.get_backbone(self.parameters.branchType)(parameters=self.parameters)

        self.frame_tensor = None  # [*height*, *width*, *channels*]
        self.target_tensor = None  # [num_targets, [*center_vertical*, *center_horizontal*, *height*, *width*]]
        self.scale_factors = None
        self.num_scales = None

        self.num_targets = None

        self.frame_size = None
        self.frame_average_colors = None
        self.frame_features_size = None

        self.frame_features = None
        self.exemplar_features = None
        self.searcharea_features = None
        self.exemplar_image = None
        self.searcharea_image = None

        self.target_in_frame_features = None
        self.target_deviation_in_searcharea_image = None
        self.target_in_searcharea_image = None
        self.target_deviation_in_searcharea_features = None
        self.target_in_searcharea_features = None

        self.exemplar_crop_target_in_frame = None
        self.exemplar_crop_target_in_frame_features = None
        self.searcharea_crop_target_in_frame = None
        self.searcharea_crop_target_in_frame_features = None

        self.frame_features_average_colors = None
        self.frame_size_represented_inside_features = None
        self.features_size_represented_inside_score = None

        self.response_size = inference_utils.get_tensor_size_after_convolution(self.parameters.searchAreaSize,
                                                              filter_size=self.parameters.exemplarSize,
                                                              stride=1, padding=0).astype("int32")


        self.virtual_exemplarSize = inference_utils.get_tensor_size_before_convolution(
            feature_size=self.parameters.exemplarSize, filter_size=self.backbone.filter_size,
            stride=self.backbone.stride, padding=self.parameters.padding)
        self.virtual_searchAreaSize = inference_utils.get_tensor_size_before_convolution(
            feature_size=self.parameters.searchAreaSize, filter_size=self.backbone.filter_size,
            stride=self.backbone.stride, padding=self.parameters.padding)


    def build(self, frame_tensor, target_tensor, frames_are_siamtf=False, scale_factors=None, shift_exemplar_size=0.0,
              shift_exemplar_center=0.0, shift_searcharea_size=0.0, shift_searcharea_center=0.0,
              name='FeatureExtractor'):
        """Builds the FeatureExtractor for the SiamMT architecture"""

        logger.debug('Creating FeatureExtractor')

        self.frames_are_siamtf = frames_are_siamtf
        self.scale_factors = scale_factors
        if self.scale_factors is None:
            self.num_scales = 1
        else:
            self.num_scales = len(self.scale_factors)

        with tf.compat.v1.variable_scope(name) as self.scope:

            # =========================== vvvvv FRAME PREPROCESSING SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('frame_preprocessing'):
                if self.parameters.standardizeInputImage:

                    if self.parameters.loadAsOpenCV:
                        self.frame_tensor = frame_tensor / 255.0
                        standardization_mean = tf.convert_to_tensor([self.input_image_standardization_mean[2],
                                                                     self.input_image_standardization_mean[1],
                                                                     self.input_image_standardization_mean[0]])
                        standardization_std = tf.convert_to_tensor([self.input_image_standardization_std[2],
                                                                    self.input_image_standardization_std[1],
                                                                    self.input_image_standardization_std[0]])
                    else:
                        standardization_mean = tf.convert_to_tensor(self.input_image_standardization_mean, dtype=tf.float32)
                        standardization_std = tf.convert_to_tensor(self.input_image_standardization_std, dtype=tf.float32)
                        self.frame_tensor = frame_tensor

                    self.frame_tensor = (self.frame_tensor - standardization_mean) / standardization_std  # height, width, channels
                else:
                    self.frame_tensor = frame_tensor  # height, width, channels

                if self.parameters.perImageStandardization:
                    self.frame_tensor = tf.image.per_image_standardization(self.frame_tensor)  # height, width, channels

                self.frame_tensor_unstandardized = frame_tensor

                self.frame_size = inference_utils.get_tensor_size(self.frame_tensor)
                self.frame_average_colors = inference_utils.get_average_channels(self.frame_tensor_unstandardized) 
            # =========================== ^^^^^ FRAME PREPROCESSING SECTION ^^^^^ ===========================

            # =========================== vvvvv TARGET PREPROCESSING SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('target_preprocessing'):
                self.target_tensor = target_tensor  # [num_targets, [center_vertical, center_horizontal, height, width]]

                self.target_center = inference_utils.get_target_center(self.target_tensor)
                self.target_size = inference_utils.get_target_size(self.target_tensor)
                self.num_targets = tf.shape(self.target_tensor)[0]

                CONTEXT_AMOUNT = self.parameters.contextAmount
                EXEMPLAR_SIZE = self.virtual_exemplarSize[0]
                w_z = self.target_size[:, 1] + CONTEXT_AMOUNT * tf.reduce_sum(self.target_size, axis=1)
                h_z = self.target_size[:, 0] + CONTEXT_AMOUNT * tf.reduce_sum(self.target_size, axis=1)
                s_z = tf.sqrt(w_z * h_z)
                scale_z = EXEMPLAR_SIZE / s_z
                self.pysot_factor = tf.stop_gradient(scale_z)

                if not self.is_training:
                    image_bbox_indexes = None
                else:
                    image_bbox_indexes = tf.range(self.num_targets, dtype=tf.int32)


            # =========================== ^^^^^ TARGET PREPROCESSING SECTION ^^^^^ ===========================

            # =========================== vvvvv SEARCHAREA/EXEMPLAR SIZE IN FRAME SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('searcharea_and_exemplar_size_in_frame'):
                self.exemplar_crop_size_in_frame, \
                self.searcharea_crop_size_in_frame = inference_utils.get_exemplar_and_searcharea_crop_sizes(
                    target_size=self.target_size,
                    exemplar_size=self.virtual_exemplarSize,
                    searcharea_size=self.virtual_searchAreaSize,
                    context_amount=self.parameters.contextAmount)

                if shift_exemplar_size > 0:
                    random_scale_in_pixels = tf.random_uniform([self.num_targets, 1],
                                                               minval=1 - shift_exemplar_size,
                                                               maxval=1 + shift_exemplar_size,
                                                               dtype=tf.float32)
                    self.exemplar_crop_size_in_frame = self.exemplar_crop_size_in_frame * random_scale_in_pixels

                self.exemplar_crop_size_in_frame = tf.stop_gradient(self.exemplar_crop_size_in_frame)

                if shift_searcharea_size > 0:
                    random_scale_in_pixels = tf.random_uniform([self.num_targets, 1],
                                                               minval=1 - shift_searcharea_size,
                                                               maxval=1 + shift_searcharea_size,
                                                               dtype=tf.float32)
                    self.searcharea_crop_size_in_frame = self.searcharea_crop_size_in_frame * random_scale_in_pixels

                self.searcharea_crop_size_in_frame = tf.stop_gradient(self.searcharea_crop_size_in_frame)


                self.searcharea_image_scale_factor = self.virtual_searchAreaSize / self.searcharea_crop_size_in_frame

                self.exemplar_crop_center_in_frame = self.target_center
                self.searcharea_crop_center_in_frame = self.target_center


                if shift_exemplar_center > 0:
                    random_displacement_in_pixels = tf.random_uniform([self.num_targets, 2],
                                                                      minval=-shift_exemplar_center,
                                                                      maxval=shift_exemplar_center,
                                                                      dtype=tf.float32)

                    random_displacement_in_pixels = random_displacement_in_pixels / self.searcharea_image_scale_factor


                    self.exemplar_crop_center_in_frame = self.exemplar_crop_center_in_frame + random_displacement_in_pixels

                self.exemplar_crop_center_in_frame = tf.stop_gradient(self.exemplar_crop_center_in_frame)


                if shift_searcharea_center > 0:
                    random_displacement_in_pixels = tf.random_uniform([self.num_targets, 2],
                                                                 minval=-shift_searcharea_center,
                                                                 maxval=shift_searcharea_center,
                                                                 dtype=tf.float32)

                    random_displacement_in_pixels = random_displacement_in_pixels / self.searcharea_image_scale_factor
                    self.searcharea_crop_center_in_frame = self.searcharea_crop_center_in_frame + random_displacement_in_pixels

                self.searcharea_crop_center_in_frame = tf.stop_gradient(self.searcharea_crop_center_in_frame)


                self.exemplar_crop_target_in_frame = tf.concat([self.exemplar_crop_center_in_frame, self.exemplar_crop_size_in_frame], axis=1)
                self.searcharea_crop_target_in_frame = tf.concat([self.searcharea_crop_center_in_frame, self.searcharea_crop_size_in_frame], axis=1)

                if self.is_training and self.frames_are_siamtf:
                    frame_axis_size = (tf.cast(self.frame_size, dtype=tf.float32) - 1.0) / 2.0

                    self.exemplar_crop_center_in_frame = tf.stop_gradient(self.exemplar_crop_center_in_frame * 0.0 + frame_axis_size)
                    self.exemplar_crop_size_in_frame = tf.stop_gradient(self.exemplar_crop_size_in_frame * 0.0 + self.virtual_exemplarSize)

                    self.searcharea_crop_center_in_frame = tf.stop_gradient(self.searcharea_crop_center_in_frame * 0.0 + frame_axis_size)
                    self.searcharea_crop_size_in_frame = tf.stop_gradient(self.searcharea_crop_size_in_frame * 0.0 + self.virtual_searchAreaSize)

                    self.exemplar_crop_target_in_frame = tf.concat(
                        [self.exemplar_crop_center_in_frame, self.exemplar_crop_size_in_frame],
                        axis=1)
                    self.searcharea_crop_target_in_frame = tf.concat(
                        [self.searcharea_crop_center_in_frame, self.searcharea_crop_size_in_frame],
                        axis=1)
            # =========================== ^^^^^ SEARCHAREA/EXEMPLAR SIZE IN FRAME SECTION ^^^^^ ===========================

            # =========================== vvvvv FRAME FEATURES SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('frame_features'):
                self.frame_features = self.backbone.build_branch(input_tensor=self.frame_tensor, is_training=self.is_training, name='siamese')

            # =========================== ^^^^^ FRAME FEATURES SECTION ^^^^^ ===========================


            # =========================== vvvvv SEARCHAREA/EXEMPLAR FEATURES CREATION SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('searcharea_and_exemplar_features_creation'):
                self.target_deviation_in_searcharea_image = self.__target_in_frame_to_searcharea_displacement(
                    target=self.target_tensor,
                    searcharea_crop_target=self.searcharea_crop_target_in_frame,
                    searcharea_output_size=self.virtual_searchAreaSize)

                self.target_in_searcharea_image = self.target_deviation_in_searcharea_image + \
                                                  tf.pad(tf.cast((self.virtual_searchAreaSize - 1) / 2,
                                                                 dtype=tf.float32),
                                                         paddings=[[0, 2]])

                exemplar_size = self.parameters.exemplarSize
                exemplar_crop_size_in_frame = self.exemplar_crop_size_in_frame

                if type(self.frame_features) in [list, tuple]:  # ESTO ES PARA LA FPN
                    min_level_stride = 2 ** (self.parameters.usedLayers[0] + 1)
                    min_level = 0
                    max_level = len(self.parameters.usedLayers) - 1
                    k_0 = self.parameters.usedLayers.index(2)

                    self.target_fpn_levels, self.target_fpn_strides = inference_utils.get_roi_levels_and_strides(
                        searcharea_crop_size_in_frame=self.searcharea_crop_size_in_frame,
                        virtual_searchAreaSize=self.virtual_searchAreaSize, k_0=k_0, min_level=min_level,
                        max_level=max_level, min_level_stride=min_level_stride)
                else:
                    self.target_fpn_levels, self.target_fpn_strides = inference_utils.get_roi_levels_and_strides(
                        searcharea_crop_size_in_frame=self.searcharea_crop_size_in_frame,
                        virtual_searchAreaSize=self.virtual_searchAreaSize,
                        k_0=0, min_level=0, max_level=0, stride_step=2, min_level_stride=8)
                    self.frame_features = [self.frame_features]

                self.exemplar_features = inference_utils.build_target_features_from_rois(
                    frame_features_layers=self.frame_features,
                    target_levels=self.target_fpn_levels,
                    target_center=self.exemplar_crop_center_in_frame,
                    frame_image_size=self.frame_size,
                    roi_crop_size=exemplar_crop_size_in_frame,
                    roi_output_size=exemplar_size)

                self.exemplar_features = inference_utils.crop_tensor_center(self.exemplar_features, crop_size=self.parameters.cropExemplarFeatures)

                self.searcharea_features = inference_utils.build_target_features_from_rois(
                    frame_features_layers=self.frame_features,
                    target_levels=self.target_fpn_levels,
                    target_center=self.searcharea_crop_center_in_frame,
                    frame_image_size=self.frame_size,
                    roi_crop_size=self.searcharea_crop_size_in_frame,
                    roi_output_size=self.parameters.searchAreaSize)

                self.exemplar_features_size = inference_utils.get_tensor_size(self.exemplar_features)
                self.searcharea_features_size = inference_utils.get_tensor_size(self.searcharea_features)

            # =========================== ^^^^^ SEARCHAREA/EXEMPLAR FEATURES CREATION SECTION ^^^^^ ===========================

            # =========================== vvvvv SEARCHAREA/EXEMPLAR IMAGE CREATION SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('searcharea_and_exemplar_image_creation'):
                self.exemplar_image = inference_utils.build_exemplar_image(
                    frame_tensor=self.frame_tensor_unstandardized,
                    target_center=inference_utils.get_target_center(self.exemplar_crop_target_in_frame),
                    exemplar_crop_size=self.exemplar_crop_size_in_frame,
                    exemplar_output_size=self.virtual_exemplarSize,
                    average_colors_tensor=self.frame_average_colors,
                    image_bbox_indexes=image_bbox_indexes,
                    method=self.parameters.tensorResizeMethod)

                if self.num_scales == 1:
                    self.searcharea_image = inference_utils.build_exemplar_image(
                        frame_tensor=self.frame_tensor_unstandardized,
                        target_center=inference_utils.get_target_center(self.searcharea_crop_target_in_frame),
                        exemplar_crop_size=self.searcharea_crop_size_in_frame,
                        exemplar_output_size=self.virtual_searchAreaSize,
                        average_colors_tensor=self.frame_average_colors,
                        image_bbox_indexes=image_bbox_indexes,
                        method=self.parameters.tensorResizeMethod)
                else:
                    self.searcharea_image = inference_utils.build_searcharea_images(
                        frame_tensor=self.frame_tensor_unstandardized,
                        target_center=inference_utils.get_target_center(self.searcharea_crop_target_in_frame),
                        searcharea_crop_size=self.searcharea_crop_size_in_frame,
                        searcharea_output_size=self.virtual_searchAreaSize,
                        average_colors_tensor=self.frame_average_colors,
                        scale_factors=self.scale_factors,
                        method=self.parameters.tensorResizeMethod)
            # =========================== ^^^^^ EXEMPLAR IMAGE CREATION SECTION ^^^^^ ===========================

            # =========================== vvvvv MISC. SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('misc'):
                self.frame_size_represented_inside_features = inference_utils.get_tensor_represented_inside_xcorr(
                    tensor_size=self.frame_size,
                    filter_size=self.backbone.filter_size,
                    stride=self.backbone.stride)
            # =========================== ^^^^^ MISC. SECTION ^^^^^ ===========================

        return self.exemplar_features, self.searcharea_features


    @staticmethod
    def __target_in_frame_to_searcharea_displacement(target, searcharea_crop_target, searcharea_output_size):
        bbox_center = inference_utils.get_target_center(target)
        bbox_size = inference_utils.get_target_size(target)

        searcharea_center = inference_utils.get_target_center(searcharea_crop_target)
        searcharea_size = inference_utils.get_target_size(searcharea_crop_target)

        searcharea_scale_factor = searcharea_output_size / searcharea_size


        new_bbox_center_displacement = bbox_center - searcharea_center

        new_bbox_center_displacement = new_bbox_center_displacement * searcharea_scale_factor
        new_bbox_size = bbox_size * searcharea_scale_factor

        new_bbox = tf.concat([new_bbox_center_displacement, new_bbox_size], axis=1)

        return new_bbox
