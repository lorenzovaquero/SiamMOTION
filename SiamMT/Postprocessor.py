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

from inference.inference_utils_tf import update_target_after_image_scaling, get_target_size, get_target_center

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2018"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzovaquero@hotmail.com"
__status__ = "Prototype"


class Postprocessor(object):

    def __init__(self):

        self.target_converted = None

    def build(self, target_tensor, frame_pad_topleft_tensor, frame_scale_factor_tensor,
              target_min_area=None, target_max_area=None, name='Postprocessor'):
        self.target_tensor = target_tensor  # [num_targets, [center_vertical, center_horizontal, height, width]]
        self.frame_pad_topleft_tensor = frame_pad_topleft_tensor
        self.frame_scale_factor_tensor = frame_scale_factor_tensor
        self.target_min_area = target_min_area
        self.target_max_area = target_max_area

        logger.debug('Creating Postprocessor')
        with tf.compat.v1.variable_scope(name):
            self.target_converted = self.target_tensor - tf.cast(tf.pad(self.frame_pad_topleft_tensor, paddings=[[0, 2]]),
                                                                 dtype=tf.float32)

            self.target_converted = update_target_after_image_scaling(self.target_converted,
                                                                      scale_factor=1 / self.frame_scale_factor_tensor)

            if self.target_min_area is not None or self.target_max_area is not None:
                target_size = get_target_size(self.target_converted)
                target_area = tf.math.reduce_prod(target_size, axis=-1)
                target_ratio = target_size[:, 0] / target_size[:, 1]

                new_target_area = target_area
                if self.target_min_area is not None:
                    new_target_area = tf.math.maximum(new_target_area, self.target_min_area)

                if self.target_max_area is not None:
                    new_target_area = tf.math.minimum(new_target_area, self.target_max_area)

                new_target_width = tf.sqrt(new_target_area / target_ratio)
                new_target_height = new_target_area / new_target_width
                new_target_size = tf.stack([new_target_height, new_target_width], axis=-1)

                self.target_converted = tf.concat([get_target_center(self.target_converted), new_target_size], axis=1)

        return self.target_converted
