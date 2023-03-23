from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import logging
import tensorflow as tf


__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


class SiameseBranch(object):
    """Network needed for extracting image features"""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.scope = None

    def restore_pretrained(self, session, ckpt_file):
        logger.info("Loading pretrained backbone.")

        alexnet_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        pretrained_saver = tf.compat.v1.train.Saver(var_list=alexnet_variables)

        pretrained_saver.restore(session, save_path=ckpt_file)

    @abc.abstractproperty
    def stride(self):
        pass

    @abc.abstractproperty
    def filter_size(self):
        pass

    @abc.abstractmethod
    def build_branch(self, input_tensor, is_training, name=None):
        pass

    def _add_receptive_field_info(self, name, kernel_size, stride_size, padding=0):
        if not hasattr(self, 'receptive_field_info'):
            self.receptive_field_info = []

        self.receptive_field_info.append({'name': name, 'kernel_size': kernel_size,
                                          'stride_size': stride_size, 'padding': padding})

    def _check_receptive_field(self):
        if not hasattr(self, 'receptive_field_info'):
            raise AttributeError('Missing `receptive_field_info` attribute! You have to call '
                                 '`_add_receptive_field_info() at each net layer.')

        logger.debug("------- vvvvv SiameseBranch net summary vvvvv -------")
        current_layer = {'name': 'IMAGE', 'tensor_size': 255, 'kernel_size': 1,
                         'stride_size': 1, 'padding': 0, 'start': 0.5}
        self.__print_layer(current_layer)
        for layer in self.receptive_field_info:
            current_layer = self.__calculate_receptive_field(layer, current_layer)
            self.__print_layer(current_layer)
        logger.debug("------- ^^^^^ SiameseBranch net summary ^^^^^ -------")

        if current_layer['kernel_size'] != self.filter_size[0]:
            raise ValueError('Global kernel size mismatch! This class declares a {} global filter size whereas '
                             'the actual kernel size is {}'.format(self.filter_size, current_layer['kernel_size']))

        if current_layer['stride_size'] != self.stride[0]:
            raise ValueError('Global stride mismatch! This class declares a {} global stride whereas the actual '
                             'stride is {}'.format(self.stride, current_layer['stride_size']))

    @classmethod
    def __calculate_receptive_field(cls, current_layer, previous_layer):
        n_in = previous_layer['tensor_size']
        j_in = previous_layer['stride_size']
        r_in = previous_layer['kernel_size']
        start_in = previous_layer['start']

        k = current_layer['kernel_size']
        s = current_layer['stride_size']
        p = current_layer['padding']

        n_out = np.floor((n_in - k + 2 * p) / s) + 1
        actualP = (n_out - 1) * s - n_in + k
        pR = np.ceil(actualP / 2)
        pL = np.floor(actualP / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - pL) * j_in
        return {'name': current_layer['name'], 'tensor_size': n_out, 'kernel_size': r_out,
                'stride_size': j_out, 'padding': p + actualP, 'start': start_out}

    @classmethod
    def __print_layer(cls, layer):
        logger.debug(layer['name'] + ":")
        logger.debug('    output size: {}'.format(int(layer['tensor_size'])))
        logger.debug('    receptive size: {}'.format(layer['kernel_size']))
        logger.debug('    global stride: {}'.format(layer['stride_size']))
        logger.debug('    accumulated padding: {}'.format(int(layer['padding'])))
        logger.debug('    kernel start: {}'.format(layer['start']))
