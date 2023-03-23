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
from inference.RPN_utils_tf import update, reshape_anchor_prediction_as_prediction, get_anchors, reshape_window_as_anchors

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class UpdateTargetRPN(object):

    def __init__(self, parameters, is_training=False):
        self.parameters = parameters
        self.is_training = is_training

        self.exemplar_features_tensor = None
        self.searcharea_features_tensor = None

        self.num_targets = None

        self.exemplar_features_size = None
        self.searcharea_features_size = None

        self.penalized_scoremap_tensor = None

        self.updated_target_in_frame = None
        self.updated_target_in_frame_features = None
        self.updated_target_in_searcharea_image = None
        self.updated_target_in_searcharea_features = None
        self.updated_target_deviation_in_searcharea_image = None

        self.penalization_window = None
        self.global_penalization_window = None


    def build(self, cls_batch_tensor, reg_batch_tensor, target_size_in_searcharea_tensor,
              searcharea_image_scale_factor_tensor, searcharea_crop_target_in_frame_tensor, frame_size_tensor,
              virtual_searchAreaSize, pysot_factor, prevent_out_of_bounds=True, min_target_size=10,
              frame_features_size_tensor=None, backbone_filter_size=None, backbone_stride=None,
              searcharea_crop_target_in_frame_features_tensor=None, searcharea_features_scale_factor_tensor=None,
              searcharea_features_size=None, name='UpdateTarget'):

        logger.debug('Creating UpdateTarget')

        self.cls_batch_tensor = cls_batch_tensor
        self.reg_batch_tensor = reg_batch_tensor
        self.target_size_in_searcharea_tensor = target_size_in_searcharea_tensor
        self.virtual_searchAreaSize = virtual_searchAreaSize
        self.searcharea_image_scale_factor_tensor = searcharea_image_scale_factor_tensor
        self.searcharea_crop_target_in_frame_tensor = searcharea_crop_target_in_frame_tensor
        self.frame_size_tensor = frame_size_tensor
        self.prevent_out_of_bounds = prevent_out_of_bounds
        self.min_target_size = min_target_size


        self.frame_features_size_tensor = frame_features_size_tensor
        self.backbone_filter_size = backbone_filter_size
        self.backbone_stride = backbone_stride
        self.searcharea_crop_target_in_frame_features_tensor = searcharea_crop_target_in_frame_features_tensor
        self.searcharea_features_scale_factor_tensor = searcharea_features_scale_factor_tensor
        self.searcharea_features_size = searcharea_features_size

        with tf.compat.v1.variable_scope(name):
            # =========================== vvvvv ANCHOR SETUP SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('anchor_setup'):
                self.response_size = inference_utils.get_tensor_size(self.cls_batch_tensor)

                response_size = inference_utils.get_tensor_size(self.cls_batch_tensor, force_numpy=True)

                self.anchors = get_anchors(total_stride=8, size=self.parameters.RPN.anchorSize,
                                           scales=self.parameters.RPN.anchorScales,
                                           ratios=self.parameters.RPN.anchorRatios,
                                           response_size=response_size)

                self.penalization_window = inference_utils.get_penalization_window(response_size,
                                                                                   method=self.parameters.windowing)

                if self.parameters.globalWindowSize > 0 and self.parameters.globalWindowInfluence > 0:
                    global_penalization_window = inference_utils.get_global_penalization_window(
                        searcharea_crop_target_in_frame=self.searcharea_crop_target_in_frame_tensor,
                        response_size=response_size,
                        frame_size=self.frame_size_tensor,
                        virtual_searcharea_size=self.virtual_searchAreaSize,
                        mask_size_multiplier=self.parameters.globalWindowSize)
                    self.global_penalization_window = tf.reduce_min(global_penalization_window, axis=0)

                    self.local_penalization_window = inference_utils.global_to_local_penalization_window(
                        global_penalization_window=global_penalization_window,
                        searcharea_crop_target_in_frame=self.searcharea_crop_target_in_frame_tensor,
                        response_size=response_size,
                        frame_size=self.frame_size_tensor)

                    # OJO! Si lo pongo con la suma pesada, solo va a peor
                    # self.penalization_window = (1 - self.parameters.globalWindowInfluence) * self.penalization_window + \
                    #                            self.parameters.globalWindowInfluence * (1 - self.local_penalization_window)

                    self.penalization_window = self.penalization_window - self.parameters.globalWindowInfluence * (1 - self.local_penalization_window)
                    self.penalization_window = tf.clip_by_value(self.penalization_window, clip_value_min=0.0, clip_value_max=float('inf'))

                    self.rpn_penalization_window = reshape_window_as_anchors(self.penalization_window,
                                                                             anchor_num=self.parameters.RPN.numAnchors)

                else:
                    self.rpn_penalization_window = reshape_window_as_anchors(self.penalization_window,
                                                                             anchor_num=self.parameters.RPN.numAnchors)

                    self.penalization_window = tf.convert_to_tensor(self.penalization_window, dtype=tf.float32)
                    self.rpn_penalization_window = tf.convert_to_tensor(self.rpn_penalization_window, dtype=tf.float32)

                    self.global_penalization_window = tf.ones([1, 1])

            # =========================== ^^^^^ ANCHOR SETUP SECTION ^^^^^ ===========================

            # =========================== vvvvv BEST PROPOSAL SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('best_proposal'):
                self.updated_target_deviation_in_crop_searcharea_image, \
                self.updated_confidence_tensor, \
                pscore = update(pred_score=self.cls_batch_tensor,
                                pred_regression=self.reg_batch_tensor,
                                previous_target_size_in_searcharea=self.target_size_in_searcharea_tensor,
                                anchors=self.anchors,
                                pysot_factor=pysot_factor,
                                rpn_penalization_window=self.rpn_penalization_window,
                                searcharea_crop_target_in_frame_tensor=self.searcharea_crop_target_in_frame_tensor,
                                window_influence=self.parameters.windowInfluence,
                                scale_and_ratio_penalty_factor=self.parameters.scaleRatioPenalty,
                                scale_damping=self.parameters.scaleDamping,
                                morph_score_flavour=self.parameters.morphScoreFlavour,
                                lr_as_pysot=self.parameters.lrAsPysot,
                                bbox_refinement=self.parameters.bboxRefinement,
                                ref_voting_iou=self.parameters.bboxRefVotingIoU,
                                ref_voting_use_pscore=self.parameters.bboxRefVotingUsePscore,
                                ref_voting_update_score=self.parameters.bboxRefVotingUpdateScore,
                                ref_top_num=self.parameters.bboxRefTopNum,
                                ref_top_use_weight=self.parameters.bboxRefTopUseWeight,
                                smooth_with_anchor_mean=self.parameters.smoothScoreWithLocationMean)

            # =========================== ^^^^^ BEST PROPOSAL SECTION ^^^^^ ===========================

            # =========================== vvvvv COORDINATES UPDATE SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('coordinates_update'):
                self.updated_target_deviation_in_searcharea_image = self.updated_target_deviation_in_crop_searcharea_image * pysot_factor
                self.updated_target_in_searcharea_image = self.updated_target_deviation_in_searcharea_image + \
                                                          tf.pad(tf.cast((self.virtual_searchAreaSize - 1) / 2,
                                                                         dtype=tf.float32),
                                                                 paddings=[[0, 2]])

                self.updated_target_in_frame = self.updated_target_deviation_in_crop_searcharea_image + \
                                               tf.pad(
                                                   inference_utils.get_target_center(self.searcharea_crop_target_in_frame_tensor),
                                                   paddings=[[0, 0], [0, 2]])

                if self.prevent_out_of_bounds:
                    self.updated_target_in_frame = inference_utils.clip_target_size(target=self.updated_target_in_frame,
                                                                                    frame_size=self.frame_size_tensor,
                                                                                    min_target_size=self.min_target_size)
            # =========================== ^^^^^ COORDINATES UPDATE SECTION ^^^^^ ===========================

            # =========================== vvvvv MISC. SECTION vvvvv ===========================
            with tf.compat.v1.variable_scope('misc'):
                self.penalized_scoremap_tensor = reshape_anchor_prediction_as_prediction(
                    pred_tensor=tf.expand_dims(pscore, axis=-1),
                    response_size=self.response_size)

                if self.frame_features_size_tensor is not None and \
                        self.backbone_filter_size is not None and \
                        self.backbone_stride is not None:
                    self.updated_target_in_frame_features = inference_utils.image_target_to_feature_target(
                        target_in_image=self.updated_target_in_frame,
                        image_size=self.frame_size_tensor,
                        feature_size=self.frame_features_size_tensor,
                        filter_size=self.backbone_filter_size,
                        stride=self.backbone_stride,
                        padding=self.parameters.padding)

                else:
                    self.updated_target_in_frame_features = None

                if self.updated_target_in_frame_features is not None and \
                        self.searcharea_crop_target_in_frame_features_tensor is not None:
                    self.updated_target_deviation_in_crop_searcharea_features = self.updated_target_in_frame_features - \
                                                                                tf.pad(inference_utils.get_target_center(
                                                                                    self.searcharea_crop_target_in_frame_features_tensor),
                                                                                    paddings=[[0, 0], [0, 2]])

                else:
                    self.updated_target_deviation_in_crop_searcharea_features = None

                if self.updated_target_deviation_in_crop_searcharea_features is not None and \
                        self.searcharea_features_scale_factor_tensor is not None:
                    self.updated_target_deviation_in_searcharea_features = self.updated_target_deviation_in_crop_searcharea_features * \
                                                                           tf.tile(self.searcharea_features_scale_factor_tensor,
                                                                                   [1, 2])

                else:
                    self.updated_target_deviation_in_searcharea_features = None

                if self.updated_target_deviation_in_searcharea_features is not None and \
                        self.searcharea_features_size is not None:
                    self.updated_target_in_searcharea_features = self.updated_target_deviation_in_searcharea_features + \
                                                                 tf.pad(tf.cast((self.searcharea_features_size - 1) / 2,
                                                                                dtype=tf.float32),
                                                                        paddings=[[0, 2]])

                else:
                    self.updated_target_in_searcharea_features = None
            # =========================== ^^^^^ MISC. SECTION ^^^^^ ===========================

        return self.updated_target_in_frame, self.updated_confidence_tensor