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

from inference.inference_utils_tf import dataset_frame_parser


__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class LoaderDataset(object):

    def __init__(self):
        self.load_as_opencv = None
        self.num_parallel_calls = None
        self.device_id = None

        self.dataset_image_path_placeholder = None

        self.dataset = None
        self.dataset_iterator = None

        self.image_tensor = None
        self.input_target = None


    def build(self, load_as_opencv=False, device_id='/gpu:0', num_parallel_calls=4, name='Loader'):
        self.load_as_opencv = load_as_opencv
        self.num_parallel_calls = num_parallel_calls
        self.device_id = device_id

        if self.load_as_opencv:
            load_as_255 = True
            load_as_bgr = True
        else:
            load_as_255 = False
            load_as_bgr = False

        logger.debug('Creating LoaderDataset')
        with tf.compat.v1.variable_scope(name):
            self.dataset_image_path_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None],
                                                                 name='Dataset_image_path_placeholder')

            self.dataset = tf.data.Dataset.from_tensor_slices(self.dataset_image_path_placeholder)

            def input_parser(image_path):
                image_decoded = dataset_frame_parser(image_path, load_as_255=load_as_255,
                                                     load_as_bgr=load_as_bgr, accurate=False)

                return image_decoded

            self.dataset = self.dataset.map(input_parser, num_parallel_calls=self.num_parallel_calls)

            self.dataset = self.dataset.apply(tf.data.experimental.prefetch_to_device(device=self.device_id, buffer_size=1))

            self.dataset_iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)

            self.image_tensor = self.dataset_iterator.get_next()

            self.input_target = tf.compat.v1.placeholder(tf.float32, [None, 4], name='target_tensor')  # [num_targets, [center_vertical, center_horizontal, height, width]]

        return self.image_tensor, self.input_target, self.dataset_image_path_placeholder, self.dataset_iterator
