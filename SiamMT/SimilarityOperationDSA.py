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

from neuralNetwork.layer_utils import convolutional_layer

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class SimilarityOperationDSA(object):
    """DSA attention for SiamAttn"""

    __fast_mode = True  # Disable during training to avoid NaNs

    def __init__(self, parameters):
        self.parameters = parameters
        self.scope = None

    def pre_build(self, exemplar_features_tensor, name='AttentionDSA'):
        """Takes as input the exemplar features and creates the necessary kernels"""

        logger.debug('PRECreating SimilarityOperationAttentionDSA')

        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as self.scope:
            # Channels attention
            exemplar_channel_attention = self._channel_attention_creation(exemplar_features_tensor)

            if self.parameters.useSelfAttention:
                # Spatial self-attention
                if hasattr(self.parameters, 'ignoreSpatialAttention') and self.parameters.ignoreSpatialAttention is True:
                    exemplar_spatial_features = 0
                else:
                    exemplar_spatial_features = self._spatial_self_attention(exemplar_features_tensor)

                # Channels self-attention
                exemplar_self_channel_features = self._channel_attention_use(exemplar_features_tensor,
                                                                             attention=exemplar_channel_attention)

                exemplar_self_attention_features = exemplar_spatial_features + exemplar_self_channel_features

            else:
                exemplar_self_attention_features = None


            if self.parameters.useCrossAttention:
                exemplar_channel_attention_kernel = exemplar_channel_attention

            else:
                exemplar_channel_attention_kernel = None


        return exemplar_self_attention_features, exemplar_channel_attention_kernel


    def build(self, exemplar_features_tensor, searcharea_features_tensor,
                    exemplar_self_attention_features=None, exemplar_channel_attention_kernel=None):
        """Takes as input the exemplar features and creates the necessary kernels"""

        logger.debug('Creating SimilarityOperationAttentionDSA')

        with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
            # Channels attention
            searcharea_channel_attention = self._channel_attention_creation(searcharea_features_tensor)

            if self.parameters.useSelfAttention:
                # Spatial self-attention
                if hasattr(self.parameters, 'ignoreSpatialAttention') and self.parameters.ignoreSpatialAttention is True:
                    searcharea_spatial_features = 0
                else:
                    searcharea_spatial_features = self._spatial_self_attention(searcharea_features_tensor)

                # Channels self-attention
                searcharea_self_channel_features = self._channel_attention_use(searcharea_features_tensor,
                                                                             attention=searcharea_channel_attention)

                searcharea_self_attention_features = searcharea_spatial_features + searcharea_self_channel_features

            else:
                searcharea_self_attention_features = None


            if self.parameters.useCrossAttention:
                exemplar_cross_attention_features = self._channel_attention_use(exemplar_features_tensor,
                                                                                attention=searcharea_channel_attention)
                searcharea_cross_attention_features = self._channel_attention_use(searcharea_features_tensor,
                                                                                  attention=exemplar_channel_attention_kernel)

            else:
                exemplar_cross_attention_features = None
                searcharea_cross_attention_features = None


            exemplar_enhanced_features = self._build_head(self_attention_features=exemplar_self_attention_features,
                                                          cross_attention_features=exemplar_cross_attention_features)

            searcharea_enhanced_features = self._build_head(self_attention_features=searcharea_self_attention_features,
                                                            cross_attention_features=searcharea_cross_attention_features)

        return exemplar_enhanced_features, searcharea_enhanced_features


    def _channel_attention_creation(self, features_tensor):
        with tf.compat.v1.variable_scope('channel', reuse=tf.compat.v1.AUTO_REUSE):
            # [?, H, W, 256]
            batch_size = tf.shape(features_tensor)[0]
            c = int(features_tensor.shape[-1])
            w = int(features_tensor.shape[-2])
            h = int(features_tensor.shape[-3])

            # C [?, 256, H, W]
            if self.__fast_mode:
                tensor = tf.cast(tf.transpose(features_tensor, [0, 3, 1, 2]), dtype=tf.float16)
            else:
                tensor = tf.cast(tf.transpose(features_tensor, [0, 3, 1, 2]), dtype=tf.float64) # To avoid infinity values

            #  [?, 256, H*W]
            proj_query = tf.reshape(tensor, [-1, c, h * w])

            # [?, H*W, 256]
            proj_key = tf.transpose(proj_query, [0, 2, 1])

            # [?, 256, 256]
            energy = tf.matmul(proj_query, proj_key)
            if not self.__fast_mode:
                energy = tf.clip_by_value(energy, clip_value_min=tf.float64.min, clip_value_max=tf.float64.max)  # To avoid infinity values

            # [?, 256, 256]
            energy_new = tf.reduce_max(energy, axis=-1, keepdims=True) - energy
            if self.__fast_mode:
                energy_new = tf.clip_by_value(energy_new, clip_value_min=tf.float16.min, clip_value_max=tf.float16.max)
            else:
                energy_new = tf.clip_by_value(energy_new, clip_value_min=tf.float64.min, clip_value_max=tf.float64.max)  # To avoid infinity values

            # [?, 256, 256]
            attention = tf.nn.softmax(energy_new, axis=-1)
            attention = tf.cast(attention, dtype=tf.float32)

            return attention

    def _channel_attention_use(self, features_tensor, attention):
        with tf.compat.v1.variable_scope('channel', reuse=tf.compat.v1.AUTO_REUSE):
            #  [?, H, W, 256]
            # [?, 256, 256]
            batch_size = tf.shape(features_tensor)[0]
            c = int(features_tensor.shape[-1])
            w = int(features_tensor.shape[-2])
            h = int(features_tensor.shape[-3])

            #  [?, 256, H, W]
            tensor = tf.transpose(features_tensor, [0, 3, 1, 2])

            if self.__fast_mode:
                attention = tf.cast(attention, dtype=tf.float16)
            else:
                attention = tf.cast(attention, dtype=tf.float64)

            # [?, 256, H*W]
            if self.__fast_mode:
                proj_value = tf.cast(tf.reshape(tensor, [-1, c, h * w]), dtype=tf.float16)
            else:
                proj_value = tf.cast(tf.reshape(tensor, [-1, c, h * w]), dtype=tf.float64)

            #  [?, 256, H*W]
            out = tf.matmul(attention, proj_value)
            if self.__fast_mode:
                out = tf.cast(tf.clip_by_value(out, clip_value_min=tf.float16.min, clip_value_max=tf.float16.max), dtype=tf.float32)
            else:
                out = tf.cast(tf.clip_by_value(out, clip_value_min=tf.float32.min, clip_value_max=tf.float32.max), dtype=tf.float32)

            # [?, 256, H, W]
            out = tf.reshape(out, [-1, c, h, w])

            # [?, H, W, 256]
            out = tf.transpose(out, [0, 2, 3, 1])

            gamma = tf.compat.v1.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.0), trainable=True)
            out = gamma * out + features_tensor

            return out


    def _spatial_self_attention(self, features_tensor):
        with tf.compat.v1.variable_scope('spatial', reuse=tf.compat.v1.AUTO_REUSE):
            #  [?, H, W, 256]
            batch_size = tf.shape(features_tensor)[0]
            c = int(features_tensor.shape[-1])
            w = int(features_tensor.shape[-2])
            h = int(features_tensor.shape[-3])

            # [?, H, W, 256]
            proj_value = convolutional_layer(features_tensor, num_filters=c, kernel_size=[1, 1],
                                             use_bias=True, weight_decay=self.parameters.convolutionWeightDecay,
                                             name='value_conv')
            # [?, 256, H, W]
            proj_value = tf.transpose(proj_value, [0, 3, 1, 2])

            # [?, 256, H*W]
            proj_value = tf.reshape(proj_value, [-1, c, h * w])


            # [?, H, W, 32]
            proj_query = convolutional_layer(features_tensor, num_filters=c // 8, kernel_size=[1, 1],
                                             use_bias=True, weight_decay=self.parameters.convolutionWeightDecay,
                                             name='query_conv')
            #  [?, 32, H, W]
            proj_query = tf.transpose(proj_query, [0, 3, 1, 2])

            # [?, 32, H*W]
            proj_query = tf.reshape(proj_query, [-1, c // 8, h * w])

            # [?, H*W, 32]
            proj_query = tf.transpose(proj_query, [0, 2, 1])


            # [?, H, W, 32]
            proj_key = convolutional_layer(features_tensor, num_filters=c // 8, kernel_size=[1, 1],
                                           use_bias=True, weight_decay=self.parameters.convolutionWeightDecay,
                                           name='key_conv')

            #  [?, 32, H, W]
            proj_key = tf.transpose(proj_key, [0, 3, 1, 2])

            # [?, 32, H*W]
            proj_key = tf.reshape(proj_key, [-1, c // 8, h * w])


            # [?, H*W, H*W]
            energy = tf.matmul(proj_query, proj_key)

            # [?, H*W, H*W]
            attention = tf.nn.softmax(energy, axis=-1)

            # [?, H*W, H*W]
            attention = tf.transpose(attention, [0, 2, 1])


            # [?, 256, H*W]
            out = tf.matmul(proj_value, attention)

            # [?, 256, H, W]
            out = tf.reshape(out, [-1, c, h, w])

            #  [?, H, W, 256]
            out = tf.transpose(out, [0, 2, 3, 1])

            alpha = tf.compat.v1.get_variable("alpha", shape=[1], initializer=tf.constant_initializer(0.0), trainable=True)
            out = alpha * out + features_tensor

            return out

    def _build_head(self, self_attention_features=None, cross_attention_features=None):
        if self_attention_features is not None and cross_attention_features is not None:
            enhanced_features = self_attention_features + cross_attention_features

        elif self_attention_features is not None and cross_attention_features is None:
            enhanced_features = self_attention_features

        elif self_attention_features is None and cross_attention_features is not None:
            enhanced_features = cross_attention_features

        elif self_attention_features is None and cross_attention_features is None:
            raise ValueError("Both `self_attention_features` and `cross_attention_features` can't be None.")

        num_channels = int(enhanced_features.shape[-1])

        enhanced_features = convolutional_layer(enhanced_features, num_filters=num_channels,
                                                kernel_size=[3, 3],
                                                use_bias=True,
                                                weight_decay=self.parameters.convolutionWeightDecay,
                                                name='deform_conv', padding='SAME')

        return enhanced_features