"""SimilarityLayer.py: Generates classification scores and regression coordinates comparing two feature maps"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import abc
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


class SimilarityLayer(object):
    """Generates a similarity value comparing two feature maps"""

    __metaclass__ = abc.ABCMeta

    variable_list = []  # Learned variables list (child classes must override this parameter)
    name = None  # Name of the layer (child classes must override this parameter)


    def __init__(self, parameters):
        self.parameters = parameters
        self.scope = None

    def restore_pretrained(self, session, ckpt_file):
        """Restores the trained variables (variable_list) stored in a checkpoint file.

                Args:
                    session: `Session` | session where the variable values will be loaded.
                    ckpt_file: `str` | path of the file containing the variable values.
        """
        logger.info("Loading pretrained Similarity Layer.")

        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        pretrained_saver = tf.compat.v1.train.Saver(var_list=var_list)

        pretrained_saver.restore(session, save_path=ckpt_file)

    @abc.abstractmethod
    def build(self, exemplar_tensor, searcharea_tensor):
        """Builds the SimilarityLayer using an exemplar and a searchaarea. Returns the similarity computation.
                Args:
                    exemplar_tensor: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels]
                        | tensor containing the object's reference. It will be transformed and used as a kernel
                        during the convolution.
                    searcharea_tensor: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels]
                        | tensor that is convoluted with the "exemplar_features", in order to search for the object.

                Returns:
                    `Tensor` (tf.float32) | containing the convolution result for each "batch".
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_kernel(self, exemplar_tensor):
        """Converts exemplar_tensor into a 'kernel', in order to speed up computations. The created kernel will be
            the input of the "build_from_kernel()" function.

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels]
                        | tensor with the object's reference that will be converted to a 'kernel' for the convolution.

                Returns:
                    `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels] | containing the exemplar
                        converted to 'kernel' format.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_from_kernel(self, exemplar_kernel, searcharea_tensor):
        """Builds the SimilarityLayer using an exemplar kernel and a searchaarea. Returns the similarity computation.
                Args:
                    exemplar_kernel: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels]
                        | tensor containing the exemplar converted to 'kernel' format..
                    searcharea_tensor: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels]
                        | tensor that is convoluted with the "exemplar_features", in order to search for the object.

                Returns:
                    `Tensor` (tf.float32) | containing the convolution result for each "batch".
        """
        raise NotImplementedError


    def _cross_correlation_training_kernel(self, exemplar_features, name='score_kernel'):
        """Training Cross-correlation Layer exemplar kernel creator
            Converts exemplar_features into a 'kernel', in order to speed up computations in
            "__cross_correlation_layer_training". Specially intended for training. The computation is "slow",
            but the backpropagation calculation is fast.

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels]
                        | tensor that will be converted to 'kernel' for the  convolution.

                Returns:
                    `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels, out_channels]
                        | containing the exemplar converted to 'kernel' format.
        """

        logger.debug('Building training cross correlation kernel')

        with tf.compat.v1.variable_scope(name):
            # ========== exemplar has shape [batch, kernel_height, kernel_width, channels, out_channels] ==========
            if len(exemplar_features.shape) == 4:
                kernel_reshape = tf.expand_dims(exemplar_features, axis=-1)

            else:
                kernel_reshape = exemplar_features

        return kernel_reshape


    def _cross_correlation_layer_training(self, exemplar_features, searcharea_features, name='score'):
        """Training Cross-correlation Layer
            Joins two siamese branches (computes a feature map using the exemplar as a filter).
            Specially intended for training. The computation is "slow", but the backpropagation calculation is fast.

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels, out_channels] ('kernel')
                        or [batch, kernel_height, kernel_width, channels] | tensor used as kernel during the convolution.
                        (if the "out_channels" dimension is missing, we assume that it's "1").
                    searcharea_features: `Tensor` (tf.float32) [batch, height, width, channels] | tensor that is
                        convoluted with the "exemplar_features".

                Returns:
                    `Tensor` (tf.float32) | containing the convolution result for each "batch" and "out_channels".
        """
        logger.debug('Building cross correlation layer')


        if len(exemplar_features.shape) == 4:
            logger.debug('Building the exemplar kernel internally, inside the "_cross_correlation_layer_training" function.')
            exemplar_features = self._cross_correlation_training_kernel(exemplar_features)


        searcharea_features = tf.expand_dims(searcharea_features, axis=1)

        with tf.compat.v1.variable_scope(name):
            score = tf.map_fn(
                lambda inputs: tf.nn.conv2d(
                    inputs[0],
                    inputs[1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    use_cudnn_on_gpu=True,
                    data_format='NHWC',
                    dilations=[1, 1, 1, 1]
                ),  # Result of conv is [1, H, W, 1]
                elems=[searcharea_features, exemplar_features],
                dtype=tf.float32
            )

            score = score[:, 0, :, :, :]  # [B, 1, H, W, nk] -> [B, H, W, nk]

            return score


    def _cross_correlation_single_searcharea_kernel(self, exemplar_features, name='score_kernel'):
        """Single-searcharea Cross-correlation Layer exemplar kernel creator
            Converts exemplar_features into a 'kernel', in order to speed up computations in
            "_cross_correlation_layer_single_searcharea"

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels, out_channels]
                        | tensor that will be converted to 'kernel' for the  convolution.

                Returns:
                    `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels] | containing the exemplar
                        converted to 'kernel' format.
        """

        logger.debug('Building single-searcharea cross correlation kernel')

        with tf.compat.v1.variable_scope(name):
            # ========== exemplar has shape [batch, kernel_height, kernel_width, channels, out_channels] ==========
            if len(exemplar_features.shape) > 4:
                try:
                    out_channels = int(tf.shape(exemplar_features)[4])
                except:
                    out_channels = -1

                if out_channels == 1:
                    kernel_reshape = exemplar_features[:, :, :, :, 0]
                else:
                    raise ValueError("Exemplar tensor must have 4 dimensions, multiple out_channels is not allowed: \"{}\"".format(str(exemplar_features)))
            else:
                kernel_reshape = exemplar_features

        return kernel_reshape


    def _cross_correlation_layer_single_searcharea(self, exemplar_features, searcharea_features, name='score'):
        """Single-searcharea Cross-correlation Layer
            Joins two siamese branches (computes a feature map using the exemplar as a filter).
            It applies different exemplar kernels to the same searcharea tensor.

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels, out_channels]
                        or [batch, kernel_height, kernel_width, channels] ('kernel') | tensor used as kernel during the
                        convolution. ("out_channels" MUST be "1" or be missing).
                    searcharea_features: `Tensor` (tf.float32) [batch, height, width, channels] | tensor that is
                        convoluted with the "exemplar_features".

                Returns:
                    `Tensor` (tf.float32) | containing the convolution result for each "batch".
        """
        logger.debug('Building single searcharea cross correlation layer')


        if len(exemplar_features.shape) > 4:
            logger.debug('Building the exemplar kernel internally, inside the "_cross_correlation_layer_single_searcharea" function.')
            exemplar_features = self._cross_correlation_single_searcharea_kernel(exemplar_features)

        #  exemplar [kernel_height, kernel_width, channels, batch]
        exemplar_features = tf.transpose(exemplar_features, [1, 2, 3, 0])

        # searcharea shape [B, H, W, C] (e.g. [?, 20, 20, 256])

        with tf.compat.v1.variable_scope(name):
            score = tf.nn.conv2d(
                searcharea_features,
                exemplar_features,
                strides=[1, 1, 1, 1],
                padding='VALID',
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                dilations=[1, 1, 1, 1],
                name=None
            )  # Result of conv is [B, H, W, kernel_batch]  # "B" suele ser 1, ya que la idea es pasarle un unico frame
            score = tf.transpose(score, [3, 1, 2, 0])  # [kernel_batch, H, W, B] 

            return score


    def _cross_correlation_depthwise_kernel(self, exemplar_features, name='score_kernel'):
        """Depthwise Cross-correlation Layer exemplar kernel creator
            Converts exemplar_features into a 'kernel', in order to speed up computations in "__cross_correlation_layer_depthwise"

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels, out_channels]
                        | tensor that will be converted to 'kernel' for the  convolution.

                Returns:
                    `Tensor` (tf.float32) [kernel_height, kernel_width, kernel_channels * batch_size, out_channels]
                        | containing the exemplar converted to 'kernel' format.
        """

        logger.debug('Building depthwise cross correlation kernel')

        with tf.compat.v1.variable_scope(name):
            # ========== exemplar has shape [batch, kernel_height, kernel_width, channels, out_channels] ==========
            batch_size = tf.shape(exemplar_features)[0]
            kernel_height = exemplar_features.shape[1]
            kernel_width = exemplar_features.shape[2]
            kernel_channels = exemplar_features.shape[3]
            out_channels = exemplar_features.shape[4]

            #  exemplar [kernel_height, kernel_width, batch, channels, out_channels]
            kernel_reshape = tf.transpose(exemplar_features, [1, 2, 0, 3, 4])
            #  exemplar [kernel_height, kernel_width, kernel_channels * batch_size, out_channels]
            kernel_reshape = tf.reshape(kernel_reshape, [kernel_height, kernel_width, kernel_channels * batch_size, out_channels])

        return kernel_reshape


    def _cross_correlation_layer_depthwise(self, exemplar_features, searcharea_features, name='score', padding="VALID"):
        """Depthwise Cross-correlation Layer
            Joins two siamese branches (computes a feature map using the exemplar as a filter).
            Specially intended for inference. The computation is "fast", but the backpropagation calculation is slow.

                Args:
                    exemplar_features: `Tensor` (tf.float32) [batch, kernel_height, kernel_width, channels, out_channels]
                        or [kernel_height, kernel_width, kernel_channels * batch_size, out_channels] ('kernel') | tensor used as
                        kernel during the convolution. (the latter dimension format is FASTER).
                    searcharea_features: `Tensor` (tf.float32) [batch, height, width, channels] | tensor that is
                        convoluted with the "exemplar_features".

                Returns:
                    `Tensor` (tf.float32) | containing the convolution result for each "batch" and "out_channels".
        """

        logger.debug('Building depthwise cross correlation layer')

        with tf.compat.v1.variable_scope(name):
            # ========== input ("searcharea") has shape [batch, input_height, input_width, channels] ==========
            input_tensor = searcharea_features
            batch_size = tf.shape(input_tensor)[0]

            input_height = input_tensor.shape[1]
            input_width = input_tensor.shape[2]
            input_channels = input_tensor.shape[3]

            input_reshape = tf.transpose(input_tensor, [1, 2, 0, 3])  # shape [input_height, input_width, batch, channels]
            input_reshape = tf.reshape(input_reshape, [1, input_height, input_width, batch_size * input_channels])

            # ========== kernel ("exemplar") has shape [kernel_height, kernel_width, kernel_channels * batch_size, out_channels] ==========
            kernel_tensor = exemplar_features

            if len(exemplar_features.shape) > 4:  #  [batch, kernel_height, kernel_width, channels, out_channels]
                logger.debug('Building the exemplar kernel internally, inside the "_cross_correlation_layer_depthwise" function.')
                # kernel has shape [batch, kernel_height, kernel_width, channels, out_channels]

                #  exemplar [kernel_height, kernel_width, kernel_channels * batch_size, out_channels]
                kernel_reshape = self._cross_correlation_depthwise_kernel(exemplar_features)

            else:
                kernel_reshape = kernel_tensor

            kernel_height = kernel_reshape.shape[0]
            kernel_width = kernel_reshape.shape[1]
            kernel_channels = tf.shape(kernel_reshape)[2] // batch_size
            out_channels = kernel_reshape.shape[3]

            score = tf.nn.depthwise_conv2d(
                input=input_reshape,
                filter=kernel_reshape,
                strides=[1, 1, 1, 1],
                padding=padding,
                name=None,
                data_format='NHWC')
            # score shape is (1, input_height-kernel_height+1, input_width-kernel_width+1, batch*channels*out_channels) if padding="VALID"
            # score shape is (1, input_height, input_width, batch*channels*out_channels) if padding="SAME"

            if padding == "VALID":
                score = tf.reshape(score, [input_height - kernel_height + 1, input_width - kernel_width + 1, batch_size, kernel_channels, out_channels])

            if padding == "SAME":
                score = tf.reshape(score, [input_height, input_width, batch_size, kernel_channels, out_channels])

            score = tf.reduce_sum(score, axis=3)  # shape [score_height, score_width, batch, out_channels]
            score = tf.transpose(score, [2, 0, 1, 3])  # shape [batch, score_height, score_width, out_channels]

            return score


    def _adjust_layer(self, input_tensor, adjust_factor, name='adjust'):
        """Adjust Layer
            Normalizes the response, in order to not saturate the sigmoid applied at the loss calculation

                Args:
                    input_tensor: `Tensor` (tf.float32) | tensor to be adjusted.
                    adjust_factor: `float` | float that will be multiplying the 'input_tensor'.
                    name: `str` | name of the layer.

                Returns:
                    `Tensor` (tf.float32) | containing the 'input_tensor' multiplied by 'adjust_factor' and
                        with a learned bias.
        """

        logger.debug('Building adjust layer')

        with tf.compat.v1.variable_scope(name):
            bias = tf.compat.v1.get_variable('bias',
                                   dtype=tf.float32,
                                   shape=[1],
                                   initializer=tf.constant_initializer(-5.878722, dtype=tf.float32),
                                   trainable=True)

            weight = tf.constant(adjust_factor,
                                 dtype=tf.float32,
                                 shape=[1],
                                 name='adjust_factor')

            layer = tf.multiply(weight, input_tensor)
            layer = tf.add(layer, bias)

            return layer