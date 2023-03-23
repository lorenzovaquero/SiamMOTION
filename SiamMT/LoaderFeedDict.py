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


__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class LoaderFeedDict(object):

    def __init__(self):
        self.input_frame_placeholder = None
        self.input_target_placeholder = None
        self.frame_tensor = None
        self.target_tensor = None

    def build(self, load_as_opencv=False, name='Loader'):
        logger.debug('Creating LoaderFeedDict')

        with tf.compat.v1.variable_scope(name):
            self.input_frame_placeholder = tf.compat.v1.placeholder(tf.uint8, [None, None, 3], name='frame_tensor')
            self.input_target_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 4], name='target_tensor')  # [num_targets, [center_vertical, center_horizontal, height, width]]

            if load_as_opencv:
                self.frame_tensor = tf.cast(self.input_frame_placeholder, dtype=tf.float32)
            else:
                self.frame_tensor = tf.reverse(self.input_frame_placeholder, axis=[-1]) 
                self.frame_tensor = tf.image.convert_image_dtype(self.frame_tensor, dtype=tf.float32)  

            self.target_tensor = self.input_target_placeholder

        return self.frame_tensor, self.target_tensor, self.input_frame_placeholder, self.input_target_placeholder
