"""inference_utils_tf.py: Utils for tracking with tensors"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import logging
import re

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


import sys
_is_python_3 = sys.version_info >= (3, 0)  # We call sys only once
if _is_python_3:
    my_str = str
else:
    my_str = basestring


def get_tensor_size(input_tensor, force_numpy=False):
    """Gets the height and width of a tensor independently of its dimensions.

    Args:
        input_tensor: `Tensor` [[batch], height, width, channels], input image or features tensor.
        force_numpy: `Bool`, whether you want the output as TF (False) or as Numpy arrays (True)

    Returns:
        `Tensor` (tf.int32) [2], with the height and width dimensions of `input_tensor`.
    """

    if len(input_tensor.shape) > 3:
        try:
            input_tensor_size = np.array([int(input_tensor.shape[1]), int(input_tensor.shape[2])], dtype=np.int32)
            if not force_numpy:
                input_tensor_size = tf.cast(input_tensor_size, dtype=tf.int32)

        except:
            input_tensor_size = tf.shape(input_tensor)[1:3]

    else:
        try:
            input_tensor_size = np.array([int(input_tensor.shape[0]), int(input_tensor.shape[1])], dtype=np.int32)
            if not force_numpy:
                input_tensor_size = tf.cast(input_tensor_size, dtype=tf.int32)

        except:
            input_tensor_size = tf.shape(input_tensor)[0:2]

    return input_tensor_size


def get_target_center(target):
    """Gets the center_vertical and center_horizontal of a target tensor.

    Args:
        target: `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], input target tensor.

    Returns:
        `Tensor` (tf.float32) [2], with the center_vertical and center_horizontal values of `target`.
    """

    if len(target.shape) == 1:
        target_center = target[0:2]
    else:
        target_center = target[:, 0:2]

    return target_center


def get_target_size(target):
    """Gets the height and width of a target tensor.

    Args:
        target: `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], input target tensor.

    Returns:
        `Tensor` (tf.float32) [2], with the height and width values of `target`.
    """

    if len(target.shape) == 1:
        target_size = target[2:4]
    else:
        target_size = target[:, 2:4]

    return target_size


def invert_target_coordinates(target):
    """Se pasa un target de [y, x, h, w] a [x, y, w, h] o viceversa"""

    if len(target.shape) == 1:
        new_target = tf.stack([target[1], target[0], target[3], target[2]], axis=0)
    else:
        new_target = tf.stack([target[:, 1], target[:, 0], target[:, 3], target[:, 2]], axis=1)

    return new_target


def update_target_after_image_scaling(target, scale_factor):
    """Calculates the new center and size of a target after the upsampling or downsampling of its image.
    Due to the target having its center as pixel positions rather than distances from the upper-left corner
    of the image, a 0.5 offset has to be considered.

    Args:
        target: `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], input target tensor.

    Returns:
        `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], with the updated `target`.
    """

    new_target_center = (get_target_center(target) + 0.5) * scale_factor - 0.5
    new_target_size = get_target_size(target) * scale_factor

    if len(target.shape) == 1:
        new_target = tf.concat([new_target_center, new_target_size], axis=0)
    else:
        new_target = tf.concat([new_target_center, new_target_size], axis=1)

    return new_target



def build_exemplar_image(frame_tensor, target_center, exemplar_crop_size, exemplar_output_size, average_colors_tensor,
                         image_bbox_indexes=None, method='bilinear'):
    """Builds an exemplar image from a frame and various targets' locations and size data"""

    logger.debug('Building exemplar image extractor')
    with tf.compat.v1.variable_scope('exemplar_crop_list_coordinates'):

        frame_size = tf.cast(get_tensor_size(frame_tensor), dtype=tf.float32)
        exemplar_axis_size = (exemplar_crop_size - 1) / 2

        # A normalized coordinate value of `y` is mapped to the image coordinate at `y * (image_height - 1)`
        area_list = tf.concat([(target_center - exemplar_axis_size) / (frame_size - 1),
                               (target_center + exemplar_axis_size) / (frame_size - 1)], axis=1)
        area_list = tf.stop_gradient(area_list)

    with tf.compat.v1.variable_scope('exemplar_crop_and_resize'):
        colored_frame = frame_tensor - average_colors_tensor  # With + and then - doesn't work
        if len(colored_frame.shape) < 4:
            colored_frame = tf.expand_dims(colored_frame, axis=0)

        if image_bbox_indexes is None:
            ind = tf.stop_gradient(tf.zeros_like(area_list[:, 0], dtype=tf.int32))
        else:
            ind = tf.stop_gradient(image_bbox_indexes)

        colored_frame_batch = tf.image.crop_and_resize(image=colored_frame, boxes=area_list, box_indices=ind,
                                                       crop_size=exemplar_output_size, method=method)

        exemplar_image_tensor = colored_frame_batch + average_colors_tensor

    return exemplar_image_tensor


def build_searcharea_images(frame_tensor, target_center, searcharea_crop_size, searcharea_output_size, average_colors_tensor,
                            scale_factors=[1.0], area_list_tensor=None, method='bilinear'):
        """Builds scaled search area images from a frame and various targets' locations and size data.
        In order to avoid gpu calculations, a precomputed area list can be passed as an argument"""

        logger.debug('Building search area image extractor')

        with tf.compat.v1.variable_scope('searcharea_crop_list_coordinates'):
            frame_size = tf.cast(get_tensor_size(frame_tensor), dtype=tf.float32)

            if area_list_tensor is None:
                tiled_scale_factors = tf.cast(tf.tile(scale_factors, [tf.shape(target_center)[0]]), dtype=tf.float32)
                tiled_scale_factors = tf.expand_dims(tiled_scale_factors, axis=-1)
                tiled_scale_factors = tf.tile(tiled_scale_factors, [1, 2])

                tiled_centers = tf.reshape(tf.tile(target_center, [1, len(scale_factors)]), [-1, 2])
                tiled_searcharea_crop_sizes = tf.reshape(tf.tile(searcharea_crop_size, [1, len(scale_factors)]), [-1, 2])

                tiled_scaled_searcharea_crop_sizes = tiled_searcharea_crop_sizes * tiled_scale_factors
                tiled_scaled_axis_sizes = (tiled_scaled_searcharea_crop_sizes - 1) / 2.0

                # A normalized coordinate value of `y` is mapped to the image coordinate at `y * (image_height - 1)`
                area_list_tensor = tf.concat([tf.math.divide(tiled_centers - tiled_scaled_axis_sizes, frame_size - 1),
                                              tf.math.divide(tiled_centers + tiled_scaled_axis_sizes, frame_size - 1)], axis=1)

        with tf.compat.v1.variable_scope('searcharea_crop_and_resize'):
            colored_frame = frame_tensor - average_colors_tensor  # With + and then - doesn't work
            if len(colored_frame.shape) < 4:
                colored_frame = tf.expand_dims(colored_frame, axis=0)

            ind = tf.zeros_like(area_list_tensor[:, 0], dtype=tf.int32)

            colored_frame_batch = tf.image.crop_and_resize(image=colored_frame, boxes=area_list_tensor, box_indices=ind,
                                                           crop_size=searcharea_output_size, method=method)

            searcharea_images_tensor = colored_frame_batch + average_colors_tensor

        return searcharea_images_tensor

def build_target_features_from_rois(frame_features_layers, target_levels, target_center, frame_image_size, roi_crop_size,
                                    roi_output_size, method='bilinear'):

    if not isinstance(frame_features_layers, list):
        raise ValueError("frame_features_layers should be a list")

    logger.debug('Building target roi extractor')
    with tf.compat.v1.variable_scope('roi_crop_list_coordinates'):

        frame_image_size = tf.cast(frame_image_size, dtype=tf.float32)
        roi_axis_size = (roi_crop_size - 1) / 2

        area_list = tf.concat([(target_center - roi_axis_size) / (frame_image_size - 1),
                               (target_center + roi_axis_size) / (frame_image_size - 1)], axis=1)
        area_list = tf.stop_gradient(area_list)

    with tf.compat.v1.variable_scope('roi_crop_and_resize'):

        if len(frame_features_layers[0].shape) == 3:
            only_one_frame = True
        elif len(frame_features_layers[0].shape) == 4:
            try:
                batch_size = int(frame_features_layers[0].shape[0])
                if batch_size == 1:
                    only_one_frame = True

                else:
                    only_one_frame = False

            except:
                only_one_frame = False

        else:
            only_one_frame = False


        if only_one_frame:
            image_indices = tf.stop_gradient(tf.zeros_like(target_levels, dtype=tf.int32))
        else:
            image_indices = tf.stop_gradient(tf.range(tf.shape(target_levels)[0], dtype=tf.int32))

        target_features = []
        target_ids = []
        for layer, layer_features in enumerate(frame_features_layers):
            target_ids_in_level = tf.where(tf.math.equal(target_levels, layer))[:, 0] 
            target_ids.append(target_ids_in_level)

            target_areas_in_level = tf.stop_gradient(tf.gather(area_list, indices=target_ids_in_level))  
            target_indices_in_level = tf.stop_gradient(tf.gather(image_indices, indices=target_ids_in_level))  

            target_features_in_level = tf.image.crop_and_resize(image=layer_features, boxes=target_areas_in_level,
                                                                box_indices=target_indices_in_level,
                                                                crop_size=roi_output_size, method=method)
            target_features.append(target_features_in_level)

        target_features = tf.concat(target_features, axis=0) 
        target_ids = tf.concat(target_ids, axis=0)  #

        index_permutation = tf.stop_gradient(tf.math.invert_permutation(target_ids)) 
        target_features = tf.gather(target_features, indices=index_permutation)
        target_ids = tf.gather(target_ids, indices=index_permutation)

    return target_features


def crop_tensor(input_tensor, top_left, bottom_right, paddings=None):
    """Crops a tensor to contain only the top_left and bottom_right points, without padding.

    Args:
        input_tensor: `Tensor` (tf.float32) [[batch], height, width, channels], input image or features tensor.
        top_left: `Tensor` (tf.float32) [2], vertical and horizontal coordinates of the top left crop point.
        bottom_right: `Tensor` (tf.float32) [2], vertical and horizontal coordinates of the bottom right crop point.
        paddings: `Tensor` (tf.int32) [2, n], padding values taken into account for the crop.

    Returns:
        `Tensor` (tf.float32) [[batch], height, width, channels], containing the points and with a size of
            bottom_right - top_left height and width.
    """

    if paddings is None:
        paddings = [0, 0]
    else:
        paddings = paddings[0:2]  # We only need the top_pad and left_pad values

    top_left_padded = tf.cast(tf.round(top_left), tf.int32) + paddings
    height_width = tf.cast(bottom_right - top_left, tf.int32)

    # crop_to_bounding_box' tensor must have either 3 or 4 dimensions
    if len(input_tensor.shape) == 2:
        input_tensor_image = tf.expand_dims(input_tensor, -1)
    else:
        input_tensor_image = input_tensor

    tensor_crop = tf.image.crop_to_bounding_box(input_tensor_image,
                                                offset_height=top_left_padded[0],
                                                offset_width=top_left_padded[1],
                                                target_height=height_width[0],
                                                target_width=height_width[1])

    # we restore the initial dimensions
    if len(input_tensor.shape) == 2:
        tensor_crop = tensor_crop[:, :, 0]

    return tensor_crop


def get_penalization_window(window_size, method='cosine'):
    """Calculates the window that will penalize displacements during tracking.

    Args:
        window_size: `Tensor` (tf.int32) [2], height and width of the resulting penalization window.
        method: `String` (str), name of the method used for distributing the weights.

    Returns:
        `Tensor` (tf.float32) [window_size[1], window_size[1]], containing the weights for penalizing displacements.
    """
    is_tensor = isinstance(window_size, (tf.Tensor, tf.SparseTensor, tf.Variable))

    if method.lower() == 'cosine'.lower():
        if is_tensor:
            # We use a raised cosine (hann) because we don't want negative points
            hann = tf.contrib.signal.hann_window(tf.reduce_max(window_size), periodic=True, dtype=tf.float32)
            hann = tf.reshape(hann, [tf.reduce_max(window_size), 1])

            # Dot product with its transpose in order to create a bidimensional cosine window
            penalization_window = tf.matmul(hann, hann, transpose_a=False, transpose_b=True)
            penalization_window = crop_tensor_center(penalization_window, window_size)

        else:
    
            penalization_window = np.outer(np.hanning(window_size[0]), np.hanning(window_size[1]))

    elif method.lower() == 'uniform'.lower():
        if is_tensor:
            penalization_window = tf.ones(window_size, dtype=tf.float32)
        else:
            penalization_window = np.ones(window_size, dtype=tf.float32)

    else:
        raise ValueError('Parameter \'windowing\' = \'' + method + '\' not supported.')

    return penalization_window


def crop_tensor_center(input_tensor, crop_size):
    """Crops a tensor at the center to a given size, without padding.

    Args:
        input_tensor: `Tensor` (tf.float32) [[batch], height, width, channels], input image or features tensor.
        crop_size: `Tensor` (tf.float32) [2], height and width of the resulting cropped tensor.

    Returns:
        `Tensor` (tf.float32) [[batch], crop_size[0], crop_size[1], channels], centered at the original input_tensor.
    """

    tensor_size = get_tensor_size(input_tensor, force_numpy=True)
    if isinstance(tensor_size, np.ndarray):
        t = (tensor_size[0] - crop_size) // 2
        b = t + crop_size

        l = (tensor_size[1] - crop_size) // 2
        r = l + crop_size
        tensor_crop = input_tensor[:, t:b, l:r, :]

    else:
        tensor_axis_size = tf.cast(tensor_size / 2, tf.float32)
        crop_axis_size = tf.cast(crop_size / 2, tf.float32)

        top_left_crop_point = tensor_axis_size - crop_axis_size
        bottom_right_crop_point = tensor_axis_size + crop_axis_size

        tensor_crop = crop_tensor(input_tensor, top_left_crop_point, bottom_right_crop_point)

    return tensor_crop


def get_average_channels(input_tensor, name='averageColors'):
    """Calculates the tensor average channel values.

    Args:
        input_tensor: `Tensor` (tf.float32) [[batch], height, width, channels], input image or features tensor.

    Returns:
        `Tensor` (tf.float32) [channels], average input_tensor channel values.
    """

    if len(input_tensor.shape) == 3:
        axis_value = (0, 1)

    elif len(input_tensor.shape) == 4:
        axis_value = (0, 1, 2)

    else:
        axis_value = None

    logger.debug('Building average channels calculation')

    average_colors = tf.reduce_mean(input_tensor, axis=axis_value, name=name)

    return average_colors


def image_target_to_feature_target(target_in_image, image_size, feature_size, filter_size, stride, padding='VALID'):
    """Calculates the center and size of a image target in feature coordinates.

    Args:
        target_in_image: `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], input target tensor
            in image coordinates.
        image_size: `Tensor` (tf.int32) [2], image tensor height and width.
        feature_size: `Tensor` (tf.int32) [2], feature tensor height and width.
        filter_size: `Number` (int32) [2], feature extractor filter height and width.
        stride: `Number` (int32) [2], feature extractor vertical and horizontal stride.

    Returns:
        `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], target tensor in feature coordinates.
    """

    target_size_in_image = get_target_size(target_in_image)  # height, width
    target_center_in_image = get_target_center(target_in_image)  # center_vertical, center_horizontal

    feature_image_factor = 1 / stride

    if padding == 'SAME':
        image_virtual_pad_topleft = 0

    elif padding == 'VALID':
        image_virtual_pad_topleft, _ = get_virtual_paddings_for_effective_size(tensor_size=image_size, filter_size=filter_size, stride=stride)
        image_virtual_pad_topleft = tf.cast(image_virtual_pad_topleft, dtype=tf.float32)

    else:
        raise ValueError('Unknown padding {}'.format(padding))

    target_center_in_virtual_image = target_center_in_image - image_virtual_pad_topleft
    target_center_in_feature = target_center_in_virtual_image * feature_image_factor

    target_size_in_feature = image_size_to_feature_size(size_in_image=target_size_in_image, image_size=image_size,
                                                        feature_size=feature_size, filter_size=filter_size,
                                                        stride=stride, padding=padding)

    if len(target_in_image.shape) == 1:
        target_in_feature = tf.concat([target_center_in_feature, target_size_in_feature], axis=0)
    else:
        target_in_feature = tf.concat([target_center_in_feature, target_size_in_feature], axis=1)

    return target_in_feature


def feature_target_to_image_target(target_in_feature, feature_size, image_size, filter_size, stride, padding='VALID'):
    """Calculates the center and size of a feature target in image coordinates.

    Args:
        target_in_feature: `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], input target
            tensor in feature coordinates.
        feature_size: `Tensor` (tf.int32) [2], feature tensor height and width.
        image_size: `Tensor` (tf.int32) [2], image tensor height and width.
        filter_size: `Number` (int32) [2], feature extractor filter height and width.
        stride: `Number` (int32) [2], feature extractor vertical and horizontal stride.

    Returns:
        `Tensor` (tf.float32) [center_vertical, center_horizontal, height, width], target tensor in image coordinates.
    """

    target_size_in_feature = get_target_size(target_in_feature)  # height, width
    target_center_in_feature = get_target_center(target_in_feature)  # center_vertical, center_horizontal

    feature_image_factor = 1 / stride

    if padding == 'SAME':
        image_virtual_pad_topleft = 0

    elif padding == 'VALID':
        image_virtual_pad_topleft, _ = get_virtual_paddings_for_effective_size(tensor_size=image_size,filter_size=filter_size, stride=stride)
        image_virtual_pad_topleft = tf.cast(image_virtual_pad_topleft, dtype=tf.float32)

    else:
        raise ValueError('Unknown padding {}'.format(padding))

    target_center_in_virtual_image = target_center_in_feature / feature_image_factor
    target_center_in_image = target_center_in_virtual_image + image_virtual_pad_topleft

    target_size_in_image = feature_size_to_image_size(size_in_feature=target_size_in_feature, image_size=image_size,
                                                      feature_size=feature_size, filter_size=filter_size,
                                                      stride=stride, padding=padding)

    if len(target_in_feature.shape) == 1:
        target_in_image = tf.concat([target_center_in_image, target_size_in_image], axis=0)
    else:
        target_in_image = tf.concat([target_center_in_image, target_size_in_image], axis=1)

    return target_in_image


def image_size_to_feature_size(size_in_image, image_size, feature_size, filter_size, stride, padding='VALID'):
    """Calculates an image size in feature coordinates.

    Args:
        size_in_image: `Tensor` (tf.float32) [height, width], input size tensor in image coordinates.
        image_size: `Tensor` (tf.int32) [2], image tensor height and width.
        feature_size: `Tensor` (tf.int32) [2], feature tensor height and width.
        filter_size: `Number` (int32) [2], feature extractor filter height and width.
        stride: `Number` (int32) [2], feature extractor vertical and horizontal stride.

    Returns:
        `Tensor` (tf.float32) [height, width], size tensor in feature coordinates.
    """

    if padding == 'SAME':
        feature_image_factor = 1 / stride

    elif padding == 'VALID':
        feature_image_factor = 1 / stride

    else:
        raise ValueError('Unknown padding {}'.format(padding))

    size_in_feature = size_in_image * feature_image_factor

    return size_in_feature


def feature_size_to_image_size(size_in_feature, feature_size, image_size, filter_size, stride, padding='VALID'):
    """Calculates a feature size in image coordinates.

    Args:
        size_in_feature: `Tensor` (tf.float32) [height, width], input size tensor in feature coordinates.
        image_size: `Tensor` (tf.int32) [2], image tensor height and width.
        feature_size: `Tensor` (tf.int32) [2], feature tensor height and width.
        filter_size: `Number` (int32) [2], feature extractor filter height and width.
        stride: `Number` (int32) [2], feature extractor vertical and horizontal stride.

    Returns:
        `Tensor` (tf.float32) [height, width], size tensor in image coordinates.
    """

    if padding == 'SAME':
        feature_image_factor = 1 / stride

    elif padding == 'VALID':
        feature_image_factor = 1 / stride

    else:
        raise ValueError('Unknown padding {}'.format(padding))

    size_in_image = size_in_feature / feature_image_factor

    return size_in_image


def get_inference_confidence(current_max_score, current_min_score, best_confidence, worse_confidence):
    confidence_max = (((current_max_score - worse_confidence) * 1.0) / (best_confidence - worse_confidence))
    confidence_min = (((current_min_score - worse_confidence) * 1.0) / (best_confidence - worse_confidence))

    current_confidence = confidence_max - confidence_min

    return current_confidence


def get_tensor_dropped_size_after_valid_convolution(tensor_size, filter_size, stride=1):
    # Tensorflow's features size (https://www.tensorflow.org/api_docs/python/tf/nn/convolution):
    features_size = get_tensor_size_after_tensorflow_convolution(tensor_size=tensor_size, filter_size=filter_size,
                                                                 stride=stride, padding='VALID')

    minimum_tensor_size = get_tensor_size_before_convolution(feature_size=features_size, filter_size=filter_size,
                                                             stride=stride, padding=0)

    dropped_size = tensor_size - minimum_tensor_size

    return dropped_size


def get_tensor_paddings_same(tensor_size, filter_size, stride):
    is_tensor = isinstance(tensor_size, (tf.Tensor, tf.SparseTensor, tf.Variable))

    # The total padding applied along the height and width is computed as:

    if is_tensor:
        out_size = tf.cast(tf.ceil(tensor_size / stride), dtype=tf.int32)
        total_pad = tf.maximum((out_size - 1) * stride + filter_size - tensor_size, 0)

    else:
        if (tensor_size % stride == 0):
            total_pad = max(filter_size - stride, 0)
        else:
            total_pad = max(filter_size - (tensor_size % stride), 0)

    # Finally, the padding on the top, bottom, left and right are:

    pad_before = total_pad // 2
    pad_after = total_pad - pad_before

    return pad_before, pad_after


def get_virtual_paddings_for_effective_size(tensor_size, filter_size, stride):
    """Cojo una imagen, le aplico un backbone que utiliza convoluciones VALID.
    Que regiones de dicha imagen serian el equivalente a padding si yo quisiera aplicar
    el mismo backbone con convoluciones SAME sobre una subseccion de la imagen
    (la de mayor tamano posible) y obtener una salida del mismo tamano a las features anteriores?"""

    image_virtual_pad_topleft = np.array([0, 0], dtype="int32")
    image_virtual_pad_bottomright = np.array([0, 0], dtype="int32")

    dropped_size = get_tensor_dropped_size_after_valid_convolution(tensor_size, filter_size=filter_size, stride=stride)
    image_virtual_pad_bottomright += dropped_size

    image_valid_size = tensor_size - dropped_size
    features_size = get_tensor_size_after_tensorflow_convolution(image_valid_size, filter_size=filter_size,
                                                                 stride=stride, padding='VALID')
    virtual_image_smallest_size = features_size * stride - (stride - 1)

    last_filter_center = (image_valid_size - 1) - tf.cast(tf.ceil((filter_size - 1) / 2), dtype=tf.int32)
    image_virtual_pad_topleft = last_filter_center - (features_size - 1) * stride  # c_f = (c_i - p) / s

    image_virtual_pad_bottomright += image_valid_size - virtual_image_smallest_size - image_virtual_pad_topleft

    return image_virtual_pad_topleft, image_virtual_pad_bottomright


def get_tensor_represented_inside_xcorr(tensor_size, filter_size, stride, get_max_value=False):
    features_size = get_tensor_size_after_tensorflow_convolution(tensor_size,
                                                                 filter_size=filter_size, stride=stride,
                                                                 padding='VALID')

    if get_max_value:
        tensor_valid_size = features_size * stride
    else:
        tensor_valid_size = features_size * stride - (stride - 1)

    return tensor_valid_size


def get_tensor_size_after_tensorflow_convolution(tensor_size, filter_size, stride=1, padding='VALID'):
    """Devuelve, no el tamano teorico (cuando las cosas encajan a la perfeccion), sino lo que hace TensorFlow"""
    is_tensor = isinstance(tensor_size, (tf.Tensor, tf.SparseTensor, tf.Variable))

    # Tensorflow's features size (https://www.tensorflow.org/api_docs/python/tf/nn/convolution):
    if padding.lower() == 'SAME'.lower():
        features_size = tensor_size / stride

    elif padding.lower() == 'VALID'.lower():
        features_size = (tensor_size - (filter_size - 1)) / stride
    else:
        raise ValueError('Unrecognized padding method "{}"'.format(padding))

    if is_tensor:
        features_size = tf.cast(tf.ceil(features_size), dtype=tf.int32)
    else:
        features_size = np.ceil(features_size).astype("int32")

    return features_size

def get_tensor_size_after_convolution(tensor_size, filter_size, stride=1, padding=0):
    if isinstance(padding, my_str):
        if padding == 'VALID':
            padding = 0
        elif padding == 'SAME':
            return tensor_size / stride

        else:
            raise ValueError('Unknown padding {}'.format(padding))

    feature_size = ((tensor_size - filter_size + 2 * padding) / stride) + 1

    return feature_size


def get_tensor_size_before_convolution(feature_size, filter_size, stride=1, padding=0):
    if isinstance(padding, my_str):
        if padding == 'VALID':
            padding = 0
        elif padding == 'SAME':
            return feature_size * stride

        else:
            raise ValueError('Unknown padding {}'.format(padding))

    tensor_size = ((feature_size - 1) * stride) + filter_size - 2 * padding

    return tensor_size


def resize_image_keep_aspect(image, new_size=None, minimum_size=None, maximum_size=None,
                             smaller_when_aspect_mismatch=True, pad_for_exact_new_size=False):
    """Resizes an image respecting its aspect ratio.
        Args:
          image: A [batch, height, width, channels] float32 tensor representing a the image/s that will be resized.
          new_size: A [2] float32 tensor representing the new height and width of the image/s.
          minimum_size: A [2] float32 tensor representing the minimum size of the new image if new_size is None.
          maximum_size: A [2] float32 tensor representing the maximum size of the new image if new_size is None.
          smaller_when_aspect_mismatch: A bool representing whether to inscribe or circunscribe the image to the size when the
            aspect ratio of the new size is different. (True = new image will always be smaller or equal than new_size).
        Returns:
          new_image: A [batch, new_height, new_width, channels] float32 tensor representing a batch of images.
          scale_factor: A [1] float32 tensor containing the scale factor used to resize the image.
        """
    image_size = get_tensor_size(image)

    new_size_bak = new_size

    if new_size is None:
        if minimum_size is None and maximum_size is None:
            raise ValueError("No sizes provided")

        if minimum_size is not None:
            minimum_size = tf.convert_to_tensor(minimum_size, dtype=tf.int32)
            new_size = tf.maximum(image_size, minimum_size)
        else:
            new_size = image_size

        if maximum_size is not None:
            maximum_size = tf.convert_to_tensor(maximum_size, dtype=tf.int32)
            new_size = tf.minimum(new_size, maximum_size)

    else:
        new_size = tf.convert_to_tensor(new_size, dtype=tf.int32)

    scale_factor_disrespecting_aspect = tf.cast(new_size, dtype=tf.float32) / tf.cast(image_size, dtype=tf.float32)

    if smaller_when_aspect_mismatch:
        scale_factor = tf.reduce_min(scale_factor_disrespecting_aspect)
    else:
        scale_factor = tf.reduce_max(scale_factor_disrespecting_aspect)

    new_size = tf.to_int32(tf.cast(image_size, dtype=tf.float32) * scale_factor)
    scale_factor = tf.cast(new_size, dtype=tf.float32) / tf.cast(image_size, dtype=tf.float32)
    new_image = tf.image.resize(image, size=new_size, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

    if new_size_bak is not None and pad_for_exact_new_size:
        if len(new_image.shape) == 3:
            paddings = [[0, new_size_bak[0] - new_size[0]], [0, new_size_bak[1] - new_size[1]], [0, 0]]
            new_shape = tf.TensorShape([new_size_bak[0], new_size_bak[1], new_image.shape[-1]])
        else:
            paddings = [[0, 0], [0, new_size_bak[0] - new_size[0]], [0, new_size_bak[1] - new_size[1]], [0, 0]]
            new_shape = tf.TensorShape([new_image.shape[0], new_size_bak[0], new_size_bak[1], new_image.shape[-1]])

        new_image = tf.pad(new_image, paddings=paddings)
        new_image.set_shape(new_shape)


    return new_image, scale_factor


def preprocess_image(inputs, invert_rgb=False, set_size=None, minimum_size=None, center_on_0=False, keep_pixel_values=False, set_exact_size=False):
    """Faster R-CNN with Inception Resnet v2 preprocessing.
    Maps pixel values to the range [-1, 1].
    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.
    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """

    scale_factor = tf.constant(1.0)

    if set_size is not None:
        inputs, scale_factor = resize_image_keep_aspect(inputs, new_size=set_size, pad_for_exact_new_size=set_exact_size)

    elif minimum_size is not None:
        inputs, scale_factor = resize_image_keep_aspect(inputs, minimum_size=minimum_size)

    if invert_rgb:
        inputs = tf.reverse(inputs, axis=[-1])

    if not keep_pixel_values:
        if center_on_0:
            inputs = (2.0 / 255.0) * tf.cast(inputs, dtype=tf.float32) - 1.0 
        else:
            inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
    else:
        inputs = tf.cast(inputs, dtype=tf.float32)

    return inputs, scale_factor


def pad_frame_for_exact_valid_convolution(frame, total_filter_size, total_stride, average_colors=None):
    if average_colors is None:
        if len(frame.shape) == 4:
            average_colors = tf.reduce_mean(frame, axis=[1, 2], name='averageColors', keepdims=True)
        else:
            average_colors = tf.reduce_mean(frame, axis=[0, 1], name='averageColors', keepdims=True)

    frame_size = get_tensor_size(frame)

    dropped_size = get_tensor_dropped_size_after_valid_convolution(tensor_size=frame_size,
                                                                   filter_size=total_filter_size,
                                                                   stride=total_stride)
    pad = (total_stride - dropped_size) % total_stride

    if len(frame.shape) == 4:
        paddings = [[0, 0], [0, pad[0]], [0, pad[1]], [0, 0]]
    else:
        paddings = [[0, pad[0]], [0, pad[1]], [0, 0]]

    frame_minus_mean = tf.cast(frame, dtype=tf.float32) - average_colors
    frame_minus_mean = tf.pad(frame_minus_mean, paddings, mode='CONSTANT', constant_values=0)
    frame = frame_minus_mean + average_colors

    return frame

def pad_frame_for_exact_same_convolution(frame, total_filter_size, total_stride, average_colors=None):
    """Toma una imagen de entrada y le aplica el pad exacto para que, si despues se le aplica un backbone con
    padding="SAME", se obtengan unas features cuyo centro sean las features que hubieramos obtenido con la imagen
    original y padding="VALID"""
    if average_colors is None:
        if len(frame.shape) == 4:
            average_colors = tf.reduce_mean(frame, axis=[1, 2], name='averageColors', keepdims=True)
        else:
            average_colors = tf.reduce_mean(frame, axis=[0, 1], name='averageColors', keepdims=True)

    frame_size = get_tensor_size(frame)

    pad_topleft, pad_bottomright = get_virtual_paddings_for_effective_size(tensor_size=frame_size,
                                                                           filter_size=total_filter_size,
                                                                           stride=total_stride)
    pad_topleft = total_stride - pad_topleft % total_stride
    pad_bottomright = total_stride - pad_bottomright % total_stride

    if len(frame.shape) == 4:
        paddings = [[0, 0], [pad_topleft[0], pad_bottomright[0]], [pad_topleft[1], pad_bottomright[1]], [0, 0]]
    else:
        paddings = [[pad_topleft[0], pad_bottomright[0]], [pad_topleft[1], pad_bottomright[1]], [0, 0]]

    frame_minus_mean = tf.cast(frame, dtype=tf.float32) - average_colors
    frame_minus_mean = tf.pad(frame_minus_mean, paddings, mode='CONSTANT', constant_values=0)
    frame = frame_minus_mean + average_colors

    return frame, pad_topleft

def pad_frame_with_effective_size(frame, total_filter_size, total_stride, average_colors=None, pad_multiple_of_stride=True):
    if average_colors is None:
        if len(frame.shape) == 4:
            average_colors = tf.reduce_mean(frame, axis=[1, 2], name='averageColors', keepdims=True)
        else:
            average_colors = tf.reduce_mean(frame, axis=[0, 1], name='averageColors', keepdims=True)

    frame_size = get_tensor_size(frame)

    pad_topleft, pad_bottomright = get_virtual_paddings_for_effective_size(tensor_size=frame_size,
                                                                           filter_size=total_filter_size,
                                                                           stride=total_stride)

    if pad_multiple_of_stride:
        additional_pad_topleft = (total_stride - (pad_topleft % total_stride)) % total_stride 
        pad_topleft = pad_topleft + additional_pad_topleft

        additional_pad_bottomright = (total_stride - ((frame_size + pad_topleft) % total_stride)) % total_stride 
        pad_bottomright = pad_bottomright + additional_pad_bottomright


    if len(frame.shape) == 4:
        paddings = [[0, 0], [pad_topleft[0], pad_bottomright[0]], [pad_topleft[1], pad_bottomright[1]], [0, 0]]
    else:
        paddings = [[pad_topleft[0], pad_bottomright[0]], [pad_topleft[1], pad_bottomright[1]], [0, 0]]

    frame_minus_mean = tf.cast(frame, dtype=tf.float32) - average_colors
    frame_minus_mean = tf.pad(frame_minus_mean, paddings, mode='CONSTANT', constant_values=0)
    frame = frame_minus_mean + average_colors

    return frame, pad_topleft



def dataset_frame_parser(image_path, load_as_255=False, load_as_bgr=False, accurate=False):
    if accurate:
        dct_method = 'INTEGER_ACCURATE'
    else:
        dct_method = ""

    image_file = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3, dct_method=dct_method)

    if load_as_bgr:
        image_decoded = tf.reverse(image_decoded, axis=[-1])

    if load_as_255:
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
    else:
        image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

    return image_decoded




def get_exemplar_and_searcharea_crop_sizes(target_size, exemplar_size, searcharea_size, context_amount):
    """Calculates the exemplar crop size and the search area crop size tensors
    Returns: float32"""

    logger.debug('Building crop sizes calculation')
    if len(target_size.shape) == 1:
        exemplar_crop_size, searcharea_crop_size = __get_exemplar_and_searcharea_crop_sizes_SINGLE(target_size, exemplar_size, searcharea_size, context_amount)
    else:
        exemplar_crop_size, searcharea_crop_size = __get_exemplar_and_searcharea_crop_sizes_MULTI(target_size, exemplar_size, searcharea_size, context_amount)

    return exemplar_crop_size, searcharea_crop_size


def __get_exemplar_and_searcharea_crop_sizes_SINGLE(target_size, exemplar_size, searcharea_size, context_amount):
    """Calculates the exemplar crop size and the search area crop size tensors
    Returns: float32"""
    is_tensor = isinstance(target_size, (tf.Tensor, tf.SparseTensor, tf.Variable))

    exemplar_aspect_ratio = exemplar_size[0] / exemplar_size[1]

    context_margin = (target_size[0] + target_size[1]) * context_amount

    if context_amount < 0:
        if is_tensor:
            exemplar_crop_size_area = tf.maximum(target_size[0] + context_margin, 1.0) * tf.maximum(target_size[1] + context_margin, 1.0)
        else:
            exemplar_crop_size_area = np.maximum(target_size[0] + context_margin, 1.0) * np.maximum(target_size[1] + context_margin, 1.0)
    else:
        exemplar_crop_size_area = (target_size[0] + context_margin) * (target_size[1] + context_margin)

    if is_tensor:
        exemplar_crop_size_width = tf.sqrt(exemplar_crop_size_area * exemplar_aspect_ratio)
    else:
        exemplar_crop_size_width = np.sqrt(exemplar_crop_size_area * exemplar_aspect_ratio)

    exemplar_crop_size_height = exemplar_crop_size_width / exemplar_aspect_ratio

    if is_tensor:
        exemplar_crop_size = tf.stack([exemplar_crop_size_width, exemplar_crop_size_height])
    else:
        exemplar_crop_size = np.stack([exemplar_crop_size_width, exemplar_crop_size_height])

    exemplar_scale_factor = exemplar_crop_size / exemplar_size
    size_difference = searcharea_size - exemplar_size
    crop_size_difference = size_difference * exemplar_scale_factor
    searcharea_crop_size = exemplar_crop_size + crop_size_difference

    return exemplar_crop_size, searcharea_crop_size


def __get_exemplar_and_searcharea_crop_sizes_MULTI(target_size, exemplar_size, searcharea_size, context_amount):
    """Calculates the exemplar crop size and the search area crop size tensors of multiple targets"""

    exemplar_aspect_ratio = exemplar_size[0] / exemplar_size[1]

    context_margin = (target_size[:, 0] + target_size[:, 1]) * context_amount

    if context_amount < 0:
        exemplar_crop_size_area = tf.maximum(target_size[:, 0] + context_margin, 1.0) * tf.maximum(target_size[:, 1] + context_margin, 1.0)
    else:
        exemplar_crop_size_area = (target_size[:, 0] + context_margin) * (target_size[:, 1] + context_margin)

    exemplar_crop_size_width = tf.sqrt(exemplar_crop_size_area * exemplar_aspect_ratio)
    exemplar_crop_size_height = exemplar_crop_size_width / exemplar_aspect_ratio

    exemplar_crop_size = tf.stack([exemplar_crop_size_width, exemplar_crop_size_height], axis=1)

    exemplar_scale_factor = exemplar_crop_size / exemplar_size
    size_difference = searcharea_size - exemplar_size
    crop_size_difference = size_difference * exemplar_scale_factor
    searcharea_crop_size = exemplar_crop_size + crop_size_difference

    return exemplar_crop_size, searcharea_crop_size


def clip_target_size(target, frame_size, min_target_size=10):
    frame_size = tf.cast(frame_size, dtype=tf.float32)
    frame_height = frame_size[0]
    frame_width = frame_size[1]

    updated_target_pos_cy = target[..., 0]
    updated_target_pos_cx = target[..., 1]
    updated_target_size_height = target[..., 2]
    updated_target_size_width = target[..., 3]

    updated_target_pos_cy = tf.clip_by_value(updated_target_pos_cy,
                                             clip_value_min=0,
                                             clip_value_max=frame_height)
    updated_target_pos_cx = tf.clip_by_value(updated_target_pos_cx,
                                             clip_value_min=0,
                                             clip_value_max=frame_width)
    updated_target_size_height = tf.clip_by_value(updated_target_size_height,
                                                  clip_value_min=min_target_size,
                                                  clip_value_max=frame_height)
    updated_target_size_width = tf.clip_by_value(updated_target_size_width,
                                                 clip_value_min=min_target_size,
                                                 clip_value_max=frame_width)

    updated_target = tf.stack([updated_target_pos_cy, updated_target_pos_cx,
                               updated_target_size_height, updated_target_size_width], axis=-1)

    return updated_target


def get_roi_levels_and_strides(searcharea_crop_size_in_frame, virtual_searchAreaSize, k_0=1, min_level=0, max_level=3, stride_step=2, min_level_stride=4, epsilon=1e-8):
    """Devuelve el nivel de la FPN en el que se deberia "recortar" el objeto, y devuelve tambien el stride de dicho nivel"""


    h = searcharea_crop_size_in_frame[..., 0]
    w = searcharea_crop_size_in_frame[..., 1]

    levels = tf.floor(k_0 + tf.log(tf.sqrt((w * h) + epsilon) / virtual_searchAreaSize[0]) / tf.log(2.)) 

    levels = tf.maximum(levels, min_level)  # level minimum is 0
    levels = tf.minimum(levels, max_level)  # level maximum is 3

    levels = tf.stop_gradient(tf.cast(tf.reshape(levels, [-1]), dtype=tf.int32))

    strides = min_level_stride * (stride_step ** levels)
    strides = tf.stop_gradient(tf.reshape(strides, [-1]))

    return levels, strides



GLOBAL_PENALIZATION_WINDOW_TEMPLATE = None  # To avoid recomputing
def get_global_penalization_window(searcharea_crop_target_in_frame, frame_size, response_size, virtual_searcharea_size,
                                   mask_size_multiplier=1.0, method='cosine'):
    global GLOBAL_PENALIZATION_WINDOW_TEMPLATE
    if GLOBAL_PENALIZATION_WINDOW_TEMPLATE is None:
        GLOBAL_PENALIZATION_WINDOW_TEMPLATE = np.expand_dims(np.expand_dims(get_penalization_window(window_size=response_size, method=method), axis=0), axis=-1)

    penalization_window_template = GLOBAL_PENALIZATION_WINDOW_TEMPLATE

    global_window_size = tf.cast(tf.ceil(response_size * (frame_size / virtual_searcharea_size)), dtype=tf.int32)

    searcharea_crop_size_in_frame = get_target_size(searcharea_crop_target_in_frame) * mask_size_multiplier
    target_scale_factor_in_frame = virtual_searcharea_size / searcharea_crop_size_in_frame

    target_in_normalized_frame = update_target_after_image_scaling(searcharea_crop_target_in_frame,
                                                                   scale_factor=target_scale_factor_in_frame)
    target_center_in_normalized_frame = get_target_center(target_in_normalized_frame)
    searcharea_axis_size_in_normalized_frame = (virtual_searcharea_size - 1) / 2
    normalized_frame_size = tf.cast(frame_size, dtype=tf.float32) * target_scale_factor_in_frame

    area_list = tf.concat([(0 - (target_center_in_normalized_frame - searcharea_axis_size_in_normalized_frame)) / tf.cast((virtual_searcharea_size - 1), dtype=tf.float32),
                           (normalized_frame_size - (target_center_in_normalized_frame - searcharea_axis_size_in_normalized_frame)) / tf.cast((virtual_searcharea_size - 1), dtype=tf.float32)],
                          axis=1)
    area_list = tf.stop_gradient(area_list)

    target_windows = tf.image.crop_and_resize(image=penalization_window_template, boxes=area_list,
                                           box_indices=tf.zeros_like(searcharea_crop_target_in_frame[:, 0], dtype=tf.int32),
                                           crop_size=global_window_size, method='bilinear', extrapolation_value=0.0)

    target_windows = target_windows[..., 0] 

    targets_mask = tf.expand_dims(1 - tf.eye(tf.shape(area_list)[0]), axis=-1)  # [?, ?, 1]
    flat_windows = tf.reshape(target_windows, shape=[1, -1, global_window_size[0] * global_window_size[1]])  # [?, 945]
    flat_windows = tf.tile(flat_windows, [tf.shape(area_list)[0], 1, 1])  # [?, ?, 945]

    target_windows = targets_mask * flat_windows
    target_windows = tf.reduce_max(target_windows, axis=1)
    target_windows = tf.reshape(target_windows, shape=[-1, global_window_size[0], global_window_size[1]])
    target_windows = 1 - target_windows

    global_penalization_window = target_windows

    return global_penalization_window


def global_to_local_penalization_window(global_penalization_window, searcharea_crop_target_in_frame, frame_size, response_size, method='bilinear'):
    crop_center = get_target_center(searcharea_crop_target_in_frame)
    crop_size_axis_size = (get_target_size(searcharea_crop_target_in_frame) - 1) / 2
    frame_size = tf.cast(frame_size, dtype=tf.float32)

    area_list = tf.concat([(crop_center - crop_size_axis_size) / (frame_size - 1),
                           (crop_center + crop_size_axis_size) / (frame_size - 1)], axis=1)
    area_list = tf.stop_gradient(area_list)

    global_penalization_window = tf.expand_dims(global_penalization_window, axis=-1)

    ind = tf.range(tf.shape(area_list)[0], dtype=tf.int32)

    local_penalization_window = tf.image.crop_and_resize(image=global_penalization_window, boxes=area_list,
                                                         box_indices=ind, crop_size=response_size, method=method)[..., 0]

    return local_penalization_window

def get_model_vars(scope=None, use_old_method=False):

    model_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        # We omit the operations created for training the network
        if '/TrainOperation/' not in v.name:
            if ('moving_mean' in v.name or 'moving_variance' in v.name) and not v.name.endswith('/ExponentialMovingAverage'):
                model_vars.append(v)

            elif ('batch_normalization' in v.name and ('/gamma' in v.name or '/beta' in v.name)) and not v.name.endswith('/ExponentialMovingAverage'):
                if use_old_method:
                    pass  # I forgot these at some point, so I have to ignore them :'D
                else:
                    model_vars.append(v)

    model_vars = list(set(model_vars))
    model_vars = sorted(model_vars, key=lambda x: x.name)

    if scope is not None:
        model_vars = [v for v in model_vars if v.name.startswith(scope)]

    return model_vars


def restore_inference_model(sess, ckpt_file, cached_variables=None, ignore_missing=False, use_old_method=False):
    sess.run(tf.global_variables_initializer())

    ckpt_variables = tf.train.list_variables(ckpt_file)
    checkpoint_has_ema = any([v[0].endswith('/ExponentialMovingAverage') for v in ckpt_variables])

    model_variables = get_model_vars(use_old_method=use_old_method)
    model_has_ema = any([v.name.endswith('/ExponentialMovingAverage') for v in tf.global_variables()])

    if ignore_missing:
        num_model_variables = len(model_variables)
        ckpt_names = [v[0] for v in ckpt_variables]
        model_variables = [v for v in model_variables if v.name in ckpt_names]
        num_found_variables = len(model_variables)

        if (num_model_variables - num_found_variables) > 0:
            print('WARNING: Couldn\'t restore {} variables'.format(num_model_variables - num_found_variables))


    if checkpoint_has_ema:
        with tf.compat.v1.variable_scope('SiamMT/TrainOperation/standard') as scope:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            var_dict = ema.variables_to_restore(model_variables)

            if cached_variables is not None:
                new_var_dict = {}
                for ckpt_var in var_dict:
                    model_var = var_dict[ckpt_var]
                    if model_var.name not in cached_variables:
                        new_var_dict[ckpt_var] = model_var

                var_dict = new_var_dict

    else:
        var_dict = model_variables

    saver = tf.train.Saver(var_dict, max_to_keep=1)
    saver.restore(sess, ckpt_file)

    if model_has_ema:
        assign_weights_to_ema(sess=sess)


def assign_weights_to_ema(sess):
    """Hace un intento de prepoblar los EMA con los valores de pesos actualmente cargados"""
    model_has_ema = any([v.name.endswith('/ExponentialMovingAverage') for v in tf.global_variables()])

    if model_has_ema:
        model_variables = get_model_vars()
        ema_suffix = '/ExponentialMovingAverage'

        model_emas = [v for v in tf.global_variables() if (v.name.endswith(ema_suffix) and '/TrainOperation/' not in v.name)]
        model_name_var_dict = {v.name: v for v in model_variables} 

        var_dict = [(model_name_var_dict[v.name[:-len(ema_suffix)]], v) for v in model_emas]
        weights_to_ema = tf.group(*(tf.assign(ema_var, var.read_value()) for var, ema_var in var_dict))
        sess.run(weights_to_ema)
