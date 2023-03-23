"""inference_utils_tf.py: Utils for tracking with tensors"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import logging

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

logger = logging.getLogger(__name__)

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


def get_anchors(total_stride, size, scales, ratios, response_size):
    """Se genera un array de dimensiones [num_ratios * num_scales * score_height * score_width, 4]
    Al recorrer dicho array de 1 en 1, primero voy variando el center_x, luego el center_y, luego la escala y luego el ratio.
    Asi, se puede ver que a esta matriz de anchors, si se le hiciera un reshape, quedaria con la forma de
    [ratio, scale, center_y, center_x, 4].
    Los anchors estan en formato [y, x, h, w]"""

    anchor_num = len(ratios) * len(scales)
    anchors = np.zeros((anchor_num, 4), dtype=np.float32)
    size = size * size
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)  # TODO: int(np.sqrt(size / ratio) * ratio)
        for scale in scales:
            anchors[count, 0] = 0
            anchors[count, 1] = 0
            anchors[count, 2] = hs * scale
            anchors[count, 3] = ws * scale
            count += 1

    anchors = np.tile(anchors, response_size[0] * response_size[1]).reshape((-1, 4))
    origin_y = - ((response_size[0] - 1) / 2) * total_stride 
    origin_x = - ((response_size[1] - 1) / 2) * total_stride 

    xs, ys = np.meshgrid([origin_x + total_stride * dx for dx in range(response_size[1])],
                         [origin_y + total_stride * dy for dy in range(response_size[0])])
    ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
    xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
    anchors[:, 0] = ys.astype(np.float32)
    anchors[:, 1] = xs.astype(np.float32)

    return anchors


def regression_values_to_bbox_coordinates(regressions, anchors):
    is_tensor = isinstance(regressions, (tf.Tensor, tf.SparseTensor, tf.Variable))

    if len(regressions.shape) == 3:
        bbox_y_center = regressions[:, :, 0] * anchors[:, 2] + anchors[:, 0]
        bbox_x_center = regressions[:, :, 1] * anchors[:, 3] + anchors[:, 1]

        if is_tensor:
            bbox_height = tf.exp(regressions[:, :, 2]) * anchors[:, 2]
            bbox_width = tf.exp(regressions[:, :, 3]) * anchors[:, 3]
            bbox = tf.stack([bbox_y_center, bbox_x_center, bbox_height, bbox_width], axis=2)

        else:
            bbox_height = np.exp(regressions[:, :, 2]) * anchors[:, 2]
            bbox_width = np.exp(regressions[:, :, 3]) * anchors[:, 3]
            bbox = np.stack([bbox_y_center, bbox_x_center, bbox_height, bbox_width], axis=2)

    else:
        bbox_y_center = regressions[:, 0] * anchors[:, 2] + anchors[:, 0]
        bbox_x_center = regressions[:, 1] * anchors[:, 3] + anchors[:, 1]
        if is_tensor:
            bbox_height = tf.exp(regressions[:, 2]) * anchors[:, 2]
            bbox_width = tf.exp(regressions[:, 3]) * anchors[:, 3]
            bbox = tf.stack([bbox_y_center, bbox_x_center, bbox_height, bbox_width], axis=1)

        else:
            bbox_height = np.exp(regressions[:, 2]) * anchors[:, 2]
            bbox_width = np.exp(regressions[:, 3]) * anchors[:, 3]
            bbox = np.stack([bbox_y_center, bbox_x_center, bbox_height, bbox_width], axis=1)

    return bbox



def get_bboxes_iou(bboxes_1, bboxes_2):
    is_tensor_1 = isinstance(bboxes_1, (tf.Tensor, tf.SparseTensor, tf.Variable))
    is_tensor_2 = isinstance(bboxes_2, (tf.Tensor, tf.SparseTensor, tf.Variable))
    """ Intersection over Union (iou) entre pares
        Args:
            bboxes_1 : [N,4]
            bboxes_2 : [K,4]
            box_type:[y,x,h,w]
        Returns:
            iou:[N,K]
    """
    #  [x1, y1, x2, y2]
    box1_y1 = bboxes_1[:, 0] - bboxes_1[:, 2] / 2
    box1_y2 = bboxes_1[:, 0] + bboxes_1[:, 2] / 2
    box1_x1 = bboxes_1[:, 1] - bboxes_1[:, 3] / 2
    box1_x2 = bboxes_1[:, 1] + bboxes_1[:, 3] / 2
    if is_tensor_1:
        box1 = tf.stack([box1_x1, box1_y1, box1_x2, box1_y2], axis=1)
        try:
            N = int(box1.get_shape()[0])
        except:
            N = tf.shape(box1)[0]

    else:
        box1 = np.stack([box1_x1, box1_y1, box1_x2, box1_y2], axis=1)
        N = box1.shape[0]

    box2_y1 = bboxes_2[:, 0] - bboxes_2[:, 2] / 2
    box2_y2 = bboxes_2[:, 0] + bboxes_2[:, 2] / 2
    box2_x1 = bboxes_2[:, 1] - bboxes_2[:, 3] / 2
    box2_x2 = bboxes_2[:, 1] + bboxes_2[:, 3] / 2
    if is_tensor_2:
        box2 = tf.stack([box2_x1, box2_y1, box2_x2, box2_y2], axis=1)
        try:
            K = int(box2.get_shape()[0])
        except:
            K = tf.shape(box2)[0]

    else:
        box2 = np.stack([box2_x1, box2_y1, box2_x2, box2_y2], axis=1)
        K = box2.shape[0]

    """ Intersection over Union (iou)
            Args:
                box1 : [N,4]
                box2 : [K,4]
                box_type:[x1,y1,x2,y2]
            Returns:
                iou:[N,K]
        """

    if is_tensor_1 or is_tensor_2:
        box1 = tf.expand_dims(box1, axis=1) + tf.zeros([1, K, 4])  # box1=[N,K,4]
        box2 = tf.expand_dims(box2, axis=0) + tf.zeros([N, 1, 4])  # box1=[N,K,4]

        x_max = tf.reduce_max(tf.stack([box1[:, :, 0], box2[:, :, 0]], axis=-1), axis=2)
        x_min = tf.reduce_min(tf.stack([box1[:, :, 2], box2[:, :, 2]], axis=-1), axis=2)
        y_max = tf.reduce_max(tf.stack([box1[:, :, 1], box2[:, :, 1]], axis=-1), axis=2)
        y_min = tf.reduce_min(tf.stack([box1[:, :, 3], box2[:, :, 3]], axis=-1), axis=2)

        tb = x_min - x_max
        lr = y_min - y_max

        tb = tf.clip_by_value(tb, 0.0, float("inf"))
        lr = tf.clip_by_value(lr, 0.0, float("inf"))

    else:
        box1 = np.expand_dims(box1, axis=1) + np.zeros([1, K, 4])  # box1=[N,K,4]
        box2 = np.expand_dims(box2, axis=0) + np.zeros([N, 1, 4])  # box1=[N,K,4]

        x_max = np.stack([box1[:, :, 0], box2[:, :, 0]], axis=-1).max(axis=2)
        x_min = np.stack([box1[:, :, 2], box2[:, :, 2]], axis=-1).min(axis=2)
        y_max = np.stack([box1[:, :, 1], box2[:, :, 1]], axis=-1).max(axis=2)
        y_min = np.stack([box1[:, :, 3], box2[:, :, 3]], axis=-1).min(axis=2)

        tb = x_min - x_max
        lr = y_min - y_max

        tb = np.clip(tb, a_min=0.0, a_max=None)
        lr = np.clip(lr, a_min=0.0, a_max=None)

    over_square = tb * lr
    all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (
            box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1]) - over_square

    return over_square / all_square



def reshape_prediction_as_anchors(pred_tensor, is_regression, N=None):
    is_tensor = isinstance(pred_tensor, (tf.Tensor, tf.SparseTensor, tf.Variable))

    if N is None:
        if is_regression:
            N = 4
        else:
            N = 2

    config_anchor_num = int(pred_tensor.shape[-1]) // N
    w_tensor = int(pred_tensor.shape[-2])
    h_tensor = int(pred_tensor.shape[-3])

    if is_tensor:
        if len(pred_tensor.shape) == 4:
            delta = pred_tensor  # [-1, H, W, A]
            delta = tf.transpose(delta, [0, 3, 1, 2])  # [-1, H, W, A] to [-1, A, H, W]
            delta = tf.reshape(delta, [-1, N, config_anchor_num * w_tensor * h_tensor])  #  [-1, A, H, W] to [-1, N, k*H*W]
            delta = tf.transpose(delta, [0, 2, 1])  #  [-1, N, k*H*W] to [-1, k*H*W, N]

        else:
            delta = pred_tensor  #  [H, W, A]
            delta = tf.transpose(delta, [2, 0, 1])  # [H, W, A] to [A, H, W]
            delta = tf.reshape(delta, [N, config_anchor_num * w_tensor * h_tensor])  #  [A, H, W] to [N, k*H*W]
            delta = tf.transpose(delta, [1, 0])  #  [N, k*H*W] to [k*H*W, N]

    else:
        if len(pred_tensor.shape) == 4:
            delta = pred_tensor  #  [-1, H, W, A]
            delta = np.transpose(delta, [0, 3, 1, 2])  # [-1, H, W, A] to [-1, A, H, W]
            delta = np.reshape(delta, [-1, N, config_anchor_num * w_tensor * h_tensor])  #  [-1, A, H, W] to [-1, N, k*H*W]
            delta = np.transpose(delta, [0, 2, 1])  #  [-1, N, k*H*W] to [-1, k*H*W, N]

        else:
            delta = pred_tensor  #  [H, W, A]
            delta = np.transpose(delta, [2, 0, 1])  # P [H, W, A] to [A, H, W]
            delta = np.reshape(delta, [N, config_anchor_num * w_tensor * h_tensor])  #  [A, H, W] to [N, k*H*W]
            delta = np.transpose(delta, [1, 0])  #  [N, k*H*W]] to [k*H*W, N]

    return delta


def reshape_anchor_prediction_as_prediction(pred_tensor, response_size):
    """Entra un tensor de la forma [?, -1, N] y sale uno de la forma [?, h, w, Nk], donde
    el -1  se corresponde con [ratio, scale, center_y, center_x] en dicho orden. Es decir, si lo
    recorremos de 1 en 1, primero iriamos variando el center_x.
    Asi, pred_regression[p_b, p_y, p_x, (p_a):(tot_a * N):tot_a]
    es equivalente a delta[p_b, p_a * tot_h * tot_w + p_y * tot_w + p_x, :]"""

    w_tensor = response_size[1]
    h_tensor = response_size[0]
    N = int(pred_tensor.shape[-1])
    config_anchor_num = int(pred_tensor.shape[1]) // (h_tensor * w_tensor)

    delta = pred_tensor  #  [-1, k*H*W, N]
    delta = tf.transpose(delta, [0, 2, 1])  #  [-1, k*H*W, N] to [-1, N, k*H*W]
    delta = tf.reshape(delta, [-1, N * config_anchor_num, h_tensor, w_tensor])  #  [-1, N, k*H*W] to [-1, A, H, W]
    delta = tf.transpose(delta, [0, 2, 3, 1])  # [-1, A, H, W] to [-1, H, W, A]

    return delta


def reshape_window_as_anchors(penalization_window, anchor_num):
    is_tensor = isinstance(penalization_window, (tf.Tensor, tf.SparseTensor, tf.Variable))

    if is_tensor:
        if len(penalization_window.shape) == 2:
            rpn_penalization_window = tf.transpose(penalization_window, [1, 0])
            rpn_penalization_window = tf.tile(tf.reshape(rpn_penalization_window, [-1]), [anchor_num])

        else:
            window_size = np.array([int(penalization_window.shape[1]), int(penalization_window.shape[2])], dtype=np.int32)
            rpn_penalization_window = tf.transpose(penalization_window, [0, 2, 1])
            rpn_penalization_window = tf.reshape(rpn_penalization_window, [-1, window_size[0] * window_size[1]])
            rpn_penalization_window = tf.tile(rpn_penalization_window, [1, anchor_num])

    else:
        if len(penalization_window.shape) == 2:
            rpn_penalization_window = penalization_window.transpose([1, 0])
            rpn_penalization_window = np.tile(np.reshape(rpn_penalization_window, [-1]), [anchor_num])

        else:
            window_size = np.array([int(penalization_window.shape[1]), int(penalization_window.shape[2])], dtype=np.int32)
            rpn_penalization_window = np.transpose(penalization_window, [0, 2, 1])
            rpn_penalization_window = np.reshape(rpn_penalization_window, [-1, window_size[0] * window_size[1]])
            rpn_penalization_window = np.tile(rpn_penalization_window, [1, anchor_num])

    return rpn_penalization_window


def get_scale_and_ratio_change_penalization(bbox_pred_size, previous_target_size, penalty_factor, context_amount=0.5, batch_first=True):
    """El previous_target_size creo que es el tamano en searcharea features"""
    is_tensor = isinstance(bbox_pred_size, (tf.Tensor, tf.SparseTensor, tf.Variable))

    def get_r_batch_first(box_size):
        r = box_size[..., 0] / box_size[..., 1]
        return r

    def get_r_batch_last(box_size):
        r = box_size[0] / box_size[1]
        return r

    def get_s_batch_first(box_size):
        p = (box_size[..., 0] + box_size[..., 1]) / 2
        s_squ = (box_size[..., 0] + p) * (box_size[..., 1] + p)
        s = tf.sqrt(s_squ)
        return s

    def get_s_batch_last(box_size):
        p = (box_size[0] + box_size[1]) / 2
        s_squ = (box_size[0] + p) * (box_size[1] + p)
        s = tf.sqrt(s_squ)
        return s

    if batch_first:
        get_r = get_r_batch_first
        get_s = get_s_batch_first
    else:
        get_r = get_r_batch_last
        get_s = get_s_batch_last

    ratio_change = tf.maximum(tf.expand_dims(get_r(previous_target_size), axis=-1) / get_r(bbox_pred_size),
                              get_r(bbox_pred_size) / tf.expand_dims(get_r(previous_target_size), axis=-1))
    scale_change = tf.maximum(tf.expand_dims(get_s(previous_target_size), axis=-1) / get_s(bbox_pred_size),
                              get_s(bbox_pred_size) / tf.expand_dims(get_s(previous_target_size), axis=-1))

    penalty = tf.exp(-(ratio_change * scale_change - 1.) * penalty_factor)

    return penalty



def bbox_voting(bboxes, scores):
    """bboxes es [batch, num_considerados, 4] mientras que scores es [batch, num_considerados, 1]"""

    numerador = tf.reduce_sum(bboxes * scores, axis=[1])
    denominador = tf.reduce_sum(scores, axis=[1])
    votos = numerador / denominador

    return votos


def iou_bbox_voting(input, iou_thresh=0.8):
  """Performs box voting as described in S. Gidaris and N.
  Komodakis, ICCV 2015.
  Performs box voting as described in 'Object detection via a multi-region &
  semantic segmentation-aware CNN model', Gidaris and Komodakis, ICCV 2015. For
  each box 'B' in selected_boxes, we find the set 'S' of boxes in pool_boxes
  with iou overlap >= iou_thresh. The location of B is set to the weighted
  average location of boxes in S (scores are used for weighting). And the score
  of B is set to the average score of boxes in S.
  Args:
    selected_boxes: BoxList containing a subset of boxes in pool_boxes. These
      boxes are usually selected from pool_boxes using non max suppression.
    pool_boxes: BoxList containing a set of (possibly redundant) boxes.
    iou_thresh: (float scalar) iou threshold for matching boxes in
      selected_boxes and pool_boxes.
  Returns:
    BoxList containing averaged locations and scores for each box in
    selected_boxes.
  Raises:
    ValueError: if
      a) if iou_thresh is not in [0, 1].
      b) pool_boxes does not have a scores field.
  """
  selected_boxes = tf.expand_dims(input[0], axis=0)
  pool_boxes = input[1]
  pool_boxes_scores = input[2]

  if not 0.0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')

  iou_ = get_bboxes_iou(selected_boxes, pool_boxes)
  match_indicator = tf.cast(tf.greater(iou_, iou_thresh), dtype=tf.float32)
  num_matches = tf.reduce_sum(match_indicator, 1)

  match_assert = tf.Assert(
      tf.reduce_all(tf.greater(num_matches, 0)), [
          'Each box in selected_boxes must match with at least one box '
          'in pool_boxes.'
      ])

  scores = tf.expand_dims(pool_boxes_scores, 1)
  scores_assert = tf.Assert(
      tf.reduce_all(tf.greater_equal(scores, 0)),
      ['Scores must be non negative.'])

  with tf.control_dependencies([scores_assert, match_assert]):
    sum_scores = tf.matmul(match_indicator, scores)
  averaged_scores = tf.reshape(sum_scores, [-1]) / num_matches

  box_locations = tf.matmul(match_indicator, pool_boxes * scores) / sum_scores
  averaged_boxes = box_locations
  return averaged_boxes, averaged_scores


def update(pred_score, pred_regression, previous_target_size_in_searcharea, anchors, rpn_penalization_window,
           searcharea_crop_target_in_frame_tensor, pysot_factor, window_influence=0.42, scale_and_ratio_penalty_factor=0.055, scale_damping=0.295, morph_score_flavour=None,
           lr_as_pysot=True, bbox_refinement=None, ref_voting_iou=0.8, ref_voting_use_pscore=False, ref_voting_update_score=False,
           ref_top_num=0, ref_top_use_weight=False, smooth_with_anchor_mean=False):

    score = reshape_prediction_as_anchors(pred_tensor=pred_score, is_regression=False)
    delta = reshape_prediction_as_anchors(pred_tensor=pred_regression, is_regression=True)

    score = tf.nn.softmax(score, axis=-1)[..., 1]

    if morph_score_flavour is not None:
        
        morph_kernel = np.ones((3, 3, 5), np.float32)
        morph_score = reshape_anchor_prediction_as_prediction(pred_tensor=tf.expand_dims(score, axis=-1),
                                                              response_size=pred_score.shape[1:3])

        if 'EROSION' in morph_score_flavour.upper():
            morph_score = tf.nn.erosion2d(value=morph_score, kernel=morph_kernel, strides=[1, 1, 1, 1],
                                             rates=[1, 1, 1, 1], padding="SAME")

        if 'DILATION' in morph_score_flavour.upper():
            morph_score = tf.nn.dilation2d(input=morph_score, filter=morph_kernel, strides=[1, 1, 1, 1],
                                              rates=[1, 1, 1, 1], padding="SAME")

        morph_score = reshape_prediction_as_anchors(pred_tensor=morph_score, is_regression=False, N=1)
        morph_score = morph_score[..., 0]
        score = morph_score

        if 'EROSION' in morph_score_flavour.upper() and 'DILATION' not in morph_score_flavour.upper():
            score += 1 

    if smooth_with_anchor_mean: 
        score = reshape_anchor_prediction_as_prediction(pred_tensor=tf.expand_dims(score, axis=-1), response_size=pred_score.shape[1:3])
        score = score * tf.reduce_mean(score, axis=-1, keepdims=True)
        score = reshape_prediction_as_anchors(pred_tensor=score, is_regression=False, N=1)
        score = score[..., 0]

    bbox_deviation_in_searcharea = regression_values_to_bbox_coordinates(regressions=delta, anchors=anchors)
    scale_penalty = get_scale_and_ratio_change_penalization(bbox_pred_size=bbox_deviation_in_searcharea[..., 2:4],
                                                      previous_target_size=previous_target_size_in_searcharea * pysot_factor,
                                                      penalty_factor=scale_and_ratio_penalty_factor)
    pscore = score * scale_penalty

    # window float
    # If rpn_penalization_window were too big, we could: `rpn_penalization_window = (rpn_penalization_window / tf.reduce_sum(rpn_penalization_window)) * tf.reduce_sum(score)`
    pscore = (1 - window_influence) * pscore + window_influence * rpn_penalization_window


    best_pscore_index = tf.argmax(pscore, axis=1, output_type=tf.int32)

    # prepare row indices
    row_indices = tf.range(tf.shape(best_pscore_index)[0])

    # zip row indices with column indices
    full_indices = tf.stack([row_indices, best_pscore_index], axis=1)

    best_pscore = None
    best_score = None

    if bbox_refinement is None or bbox_refinement.upper() == '':
        # retrieve values by indices
        best_bbox_deviation_in_searcharea = tf.gather_nd(bbox_deviation_in_searcharea, full_indices)
        best_bbox_deviation_in_searcharea = best_bbox_deviation_in_searcharea / pysot_factor

    elif bbox_refinement.upper() == 'VOTING':
        bbox_deviation_in_frame = bbox_deviation_in_searcharea / tf.expand_dims(pysot_factor, axis=-1)  # [N, 3125, 4]

        if ref_voting_use_pscore:
            bbox_pool_scores = pscore
        else:
            bbox_pool_scores = score

        best_bbox_deviation_in_frame = tf.gather_nd(bbox_deviation_in_frame, full_indices)  # [N, 4]

        def box_voting(inputs):
            return iou_bbox_voting(inputs, iou_thresh=ref_voting_iou)

        voting_output = tf.vectorized_map(box_voting,
                                          (best_bbox_deviation_in_frame, bbox_deviation_in_frame, bbox_pool_scores))
        voted_bboxes = voting_output[0]  # [N, 1, 4]
        voted_scores = voting_output[1]  # [N, 1]

        best_bbox_deviation_in_searcharea = voted_bboxes[:, 0, :]  # [N, 4]
        voted_scores = voted_scores[:, 0]  # [N]

        if ref_voting_update_score:
            if ref_voting_use_pscore:
                best_pscore = voted_scores
            else:
                best_score = voted_scores

    elif bbox_refinement.upper() == 'TOP_K':
        assert ref_top_num >= 2, "top_k refinement chosen but without specifying the number of considered bboxes"

        best_pscore_value_refinement, best_pscore_index_refinement = tf.nn.top_k(pscore, k=ref_top_num, sorted=True)

        best_pscore_index_refinement_vector = tf.reshape(best_pscore_index_refinement, [-1])

        # prepare row indices
        row_indices = tf.range(tf.shape(best_pscore_index_refinement)[0])
        row_indices = tf.expand_dims(row_indices, axis=-1)  # Convert to a num_targets x 1 matrix.
        row_indices = tf.tile(row_indices, [1, ref_top_num])  # Create multiple columns.
        row_indices = tf.reshape(row_indices, [-1])  # Convert back to a vector.

        # zip row indices with column indices
        full_indices_refinement = tf.stack([row_indices, best_pscore_index_refinement_vector], axis=1)

        # retrieve values by indices
        best_bbox_deviation_in_searcharea_candidates = tf.gather_nd(bbox_deviation_in_searcharea,
                                                                    full_indices_refinement)
        best_bbox_deviation_in_searcharea_candidates = tf.reshape(best_bbox_deviation_in_searcharea_candidates,
                                                                  [-1, ref_top_num, 4])

        if ref_top_use_weight:
            best_pscore_pred = tf.gather_nd(pscore, full_indices_refinement)
            best_pscore_pred = tf.reshape(best_pscore_pred, [-1, ref_top_num, 1])

            best_bbox_deviation_in_searcharea = bbox_voting(bboxes=best_bbox_deviation_in_searcharea_candidates,
                                                            scores=best_pscore_pred)

        else:
            best_bbox_deviation_in_searcharea = tf.reduce_mean(best_bbox_deviation_in_searcharea_candidates, axis=[1])

        best_bbox_deviation_in_searcharea = best_bbox_deviation_in_searcharea / pysot_factor 

    else:
        raise ValueError('Unsupported bbox voting type {}'.format(bbox_refinement))

    if best_pscore is None: 
        best_pscore = tf.gather_nd(pscore, full_indices)

    if best_score is None: 
        best_score = tf.gather_nd(score, full_indices)

    best_scale_penalty = tf.gather_nd(scale_penalty, full_indices)

    if lr_as_pysot:
        lr = scale_damping * best_score * best_scale_penalty
    else:
        lr = scale_damping * best_pscore

    res_0 = best_bbox_deviation_in_searcharea[:, 0]
    res_1 = best_bbox_deviation_in_searcharea[:, 1]
    res_2 = previous_target_size_in_searcharea[:, 0] * (1 - lr) + best_bbox_deviation_in_searcharea[:, 2] * lr
    res_3 = previous_target_size_in_searcharea[:, 1] * (1 - lr) + best_bbox_deviation_in_searcharea[:, 3] * lr

    updated_target_deviation_in_searcharea = tf.stack([res_0, res_1, res_2, res_3], axis=-1)
    # [y, x, h, w]

    return updated_target_deviation_in_searcharea, best_pscore, pscore

