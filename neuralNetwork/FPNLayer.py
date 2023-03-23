"""FPNLayer.py: Generates FPN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

slim = tf.contrib.slim

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"

class FPNLayer(object):
    """Generates a FPN"""

    name = 'FPN'

    def __init__(self, parameters):
        self.parameters = parameters
        self.scope = None

    def build(self, input_tensor_list, is_training, create_top_level=False, num_output_channels=256, scope_name=name):

        # input_tensor_list = [C2, C3, C4, C5]

        logger.debug('Building FPN layer \'' + scope_name + '\'')

        pyramid_list = [None] * len(input_tensor_list)

        with tf.compat.v1.variable_scope(scope_name) as self.scope:
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.parameters.convolutionWeightDecay),
                                activation_fn=None, normalizer_fn=None):

                last_level = len(input_tensor_list)+1  # From P2 to P5
                last_level_name = 'P{}'.format(last_level)

                P5 = slim.conv2d(input_tensor_list[-1],  # C5
                                 num_outputs=num_output_channels,
                                 kernel_size=[1, 1],
                                 stride=1, scope='build_' + last_level_name)

                if create_top_level:
                    P6 = slim.max_pool2d(P5, kernel_size=[1, 1], stride=2, scope='build_' + 'P{}'.format(last_level+1))
                    pyramid_list.append(P6)

                pyramid_list[last_level-2] = P5 

                for level in range(len(input_tensor_list), 0, -1):  # build [P4, P3, P2]
                    list_index = level - 2
                    pyramid_list[list_index] = self.fusion_two_layer(C_i=input_tensor_list[list_index],
                                                                     P_j=pyramid_list[list_index+1],
                                                                     scope='build_' + 'P{}'.format(level),
                                                                     num_output_channels=num_output_channels)

                for level in range(len(input_tensor_list), 0, -1):
                    list_index = level - 2
                    pyramid_list[list_index] = slim.conv2d(pyramid_list[list_index],
                                                           num_outputs=num_output_channels, kernel_size=[3, 3],
                                                           padding="SAME", stride=1, scope='fuse_' + 'P{}'.format(level))

            return pyramid_list

    @staticmethod
    def fusion_two_layer(C_i, P_j, scope, num_output_channels=256):
        '''
        i = j+1
        :param C_i: shape is [1, h, w, c]
        :param P_j: shape is [1, h/2, w/2, 256]
        :return:
        P_i
        '''
        with tf.variable_scope(scope):
            level_name = scope.split('_')[1]
            h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
            upsample_p = tf.image.resize_bilinear(P_j,
                                                  size=[h, w],
                                                  name='up_sample_' + level_name)

            reduce_dim_c = slim.conv2d(C_i,
                                       num_outputs=num_output_channels,
                                       kernel_size=[1, 1], stride=1,
                                       scope='reduce_dim_' + level_name)

            add_f = 0.5 * upsample_p + 0.5 * reduce_dim_c

            # P_i = slim.conv2d(add_f,
            #                   num_outputs=256, kernel_size=[3, 3], stride=1,
            #                   padding='SAME',
            #                   scope='fusion_'+level_name)
            return add_f