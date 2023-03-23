from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import logging


from .SiameseBranch import SiameseBranch
from .NeckLayer import NeckLayer
from .FPNLayer import FPNLayer
from .layer_utils import convolutional_layer, batch_normalization_layer, relu_layer, max_pooling_layer

logger = logging.getLogger(__name__)

slim = tf.contrib.slim

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class SiameseBranch_ResNet(SiameseBranch):
    """Network needed for extracting image features. A ResNet-based network"""

    __metaclass__ = abc.ABCMeta

    stride = np.array([8, 8])

    __is_emulating_PySOT = False

    @abc.abstractproperty
    def flavour(self):
        pass



    def __init__(self, parameters):
        super(SiameseBranch_ResNet, self).__init__()
        self.parameters = parameters
        self.blocks = None
        self.block_type = None

        self.all_layers = None
        self.used_layers = None
        self.adjusted_layers = None

        if self.flavour is None or self.flavour == '':
            raise ValueError('Please, select a ResNet flavour')

        elif self.flavour == '18':
            self.blocks = [2, 2, 2, 2]
            self.block_type = 'basic'

        elif self.flavour == '34':
            self.blocks = [3, 4, 6, 3]
            self.block_type = 'basic'

        elif self.flavour == '50':
            self.blocks = [3, 4, 6, 3]
            self.block_type = 'bottleneck'

        elif self.flavour == '101':
            self.blocks = [3, 4, 23, 3]
            self.block_type = 'bottleneck'

        elif self.flavour == '152':
            self.blocks = [3, 8, 36, 3]
            self.block_type = 'bottleneck'

        else:
            raise ValueError('Unknown ResNet flavour {}'.format(self.flavour))


        if self.block_type.lower() == 'Basic'.lower():
            self.block_expansion = 1

        elif self.block_type.lower() == 'Bottleneck'.lower():
            self.block_expansion = 4

        else:
            raise ValueError("Unrecognized block type {}".format(self.block_type))

        if self.parameters.adjust is None or self.parameters.adjust == "":
            self.adjuster_layer = None

        elif self.parameters.adjust.lower() == "neck".lower():
            self.adjuster_layer = NeckLayer(self.parameters)

        elif self.parameters.adjust.lower() == "fpn".lower():
            self.adjuster_layer = FPNLayer(self.parameters)

        else:
            raise ValueError("Unrecognized adjust type {}".format(self.parameters.adjust))


    def build_branch(self, input_tensor, is_training, name='siamese'):
        """Builds the part of a ResNet that is needed for extracting image features

        Args:
            input_tensor: `Tensor`, input image tensor.
            is_training: `bool`, whether the branch will be used for training.
            name: `string`, name of the branch.

        Returns:
            `Tensor` with the batch dimension of `input_tensor`, containing its extracted image features.

        Example:
            >>> import numpy as np
            >>> blue_channel = np.full((127, 127), 0.2)
            >>> green_channel = np.full((127, 127), 0.7)
            >>> red_channel = np.full((127, 127), 0.1)
            >>> image = np.stack((blue_channel, green_channel, red_channel), axis=2)
            >>> self.build_branch(image, is_training=True, name='exemplar')
            [extracted image features]
        """

        logger.debug('Building ResNet branch \'' + name + '\'')

        self._is_training = is_training


        with tf.compat.v1.variable_scope(name, 'siamese', values=[input_tensor], reuse=tf.compat.v1.AUTO_REUSE) as self.scope:

            x = self._ConvNormRelu(input_tensor, num_channels=64, kernel_size=7, stride=2, name='', op_surname='1')
            self.conv1 = x

            paddings = [[1, 1], [1, 1], [0, 0]]
            if len(x.shape) == 4:
                paddings.insert(0, [0, 0])

            x = tf.pad(x, paddings=paddings, name='.pad')
            x = max_pooling_layer(x, kernel_size=[3, 3], stride=2, name='maxpool', padding='VALID') 

            x = self._Layer(x, num_channels=64, blocks=self.blocks[0], stride=1, name='layer1')
            self.layer1 = x

            x = self._Layer(x, num_channels=128, blocks=self.blocks[1], stride=2, name='layer2')
            self.layer2 = x

            if self.parameters.useDilation:
                layer3_dilation = 2
                layer3_stride = 1
                layer4_dilation = 4
                layer4_stride = 1
            else:
                layer3_dilation = 1
                layer3_stride = 2
                layer4_dilation = 1
                layer4_stride = 2

            x = self._Layer(x, num_channels=256, blocks=self.blocks[2], stride=layer3_stride, dilation=layer3_dilation, name='layer3')
            self.layer3 = x

            x = self._Layer(x, num_channels=512, blocks=self.blocks[3], stride=layer4_stride, dilation=layer4_dilation, name='layer4')
            self.layer4 = x

            self.all_layers = [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4]

            if self.parameters.usedLayers is None:
                self.used_layers = x

            elif type(self.parameters.usedLayers) in [list, tuple]:
                self.used_layers = [self.all_layers[i] for i in self.parameters.usedLayers]

            elif isinstance(self.parameters.usedLayers, int):
                self.used_layers = self.all_layers[self.parameters.usedLayers]

            else:
                raise ValueError('Unsupported usedLayers {} ({})'.format(self.parameters.usedLayers, type(self.parameters.usedLayers)))

            if self.adjuster_layer is None:
                self.adjusted_layers = self.used_layers

            else:
                self.adjusted_layers = self.adjuster_layer.build(self.used_layers, is_training=self._is_training)

        return self.adjusted_layers


    def _Conv(self, inputs, num_channels, kernel_size, stride=1, dilation=1, padding=None, name='conv'):
        """The Pytorch model uses same padding but naively setting the padding to same in Tensorflow will not work in some
        layers. I did not comprehensively study the cases where it results in a different output but it seems to occur when
        stride>1. To get around this we insert a zero padding layer prior to our strided layer where we now use valid
        padding. For example for a max pool layer we can do the following"""
        padding_type = 'VALID'

        if padding is None: 
            if self.parameters.padding.lower() == 'VALID'.lower(): 
                padding = 0
                if kernel_size > 1:
                    padding = 2 - stride
                if dilation > 1:
                    padding = dilation

            elif self.parameters.padding.lower() == 'SAME'.lower():
                padding = 0

                if stride > 1:
                    padding_type = 'VALID'
                    padding = (kernel_size - 1) // 2

                else:
                    padding_type = 'SAME'

                if dilation > 1:
                    padding_type = 'VALID'
                    padding = dilation

            else:
                raise ValueError('Unrecognized padding type {}'.format(self.parameters.padding))

        assert stride == 1 or dilation == 1, "stride must be 1 if dilation_rate > 1"

        if padding > 0:
            paddings = [[padding, padding], [padding, padding], [0, 0]]
            if len(inputs.shape) == 4:
                paddings.insert(0, [0, 0])

            x = tf.pad(inputs, paddings=paddings, name=name + '.pad')

        else:
            x = inputs

        x = convolutional_layer(x, num_filters=num_channels, kernel_size=kernel_size, stride=stride,
                                dilation_rate=dilation, padding=padding_type, use_bias=False,
                                weight_decay=self.parameters.convolutionWeightDecay,
                                biases_decay=self.parameters.convolutionBiasDecay,
                                init_method=self.parameters.initMethod, name=name)
        pad_str = ', padding=({}, {})'.format(padding, padding) if padding > 0 else ''
        dilation_str = ', dilation=({}, {})'.format(dilation, dilation) if dilation > 1 else ''

        return x

    def _ConvNormRelu(self, inputs, num_channels, kernel_size, stride=1, dilation=1, name='', op_surname='1'):
        """Conv-BatchNorm-ReLU block"""
        if name == '':
            pass
        else:
            name = name + '.'


        x = self._Conv(inputs, num_channels=num_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                       name=name + 'conv' + op_surname)
        x = batch_normalization_layer(x, is_training=self._is_training,
                                      momentum=self.parameters.batchNormalizationWeightDecay,
                                      epsilon=self.parameters.epsilon,
                                      use_scale=True, name=name + 'bn' + op_surname)
        x = relu_layer(x, name=name + 'relu' + op_surname)

        return x


    def _block_section(self, inputs, num_channels, stride=1, dilation=1, name=''):
        if self.block_type.lower() == 'Basic'.lower():
            x = self._basic_section(inputs, num_channels=num_channels, stride=stride, dilation=dilation, name=name)

        elif self.block_type.lower() == 'Bottleneck'.lower():
            x = self._bottleneck_section(inputs, num_channels=num_channels, stride=stride, dilation=dilation, name=name)

        else:
            raise ValueError("Unrecognized block type {}".format(self.block_type))

        return x


    def _basic_section(self, inputs, num_channels, stride=1, dilation=1, name=''):
        dd = dilation
        if dilation > 1:
            if int(inputs.shape[-1]) == num_channels:
                dd = dilation
            else:
                dd = dilation // 2

        x = self._ConvNormRelu(inputs, num_channels=num_channels, kernel_size=3, stride=stride, dilation=dd, name=name, op_surname='1')
        x = self._Conv(x, num_channels=num_channels, kernel_size=3, stride=1, dilation=dilation, name=name + '.conv2')
        x = batch_normalization_layer(x, is_training=self._is_training,
                                      momentum=self.parameters.batchNormalizationWeightDecay,
                                      epsilon=self.parameters.epsilon,
                                      use_scale=True, name=name + '.bn2')
        return x

    def _bottleneck_section(self, inputs, num_channels, stride=1, dilation=1, name=''):
        x = self._ConvNormRelu(inputs, num_channels=num_channels, kernel_size=1, stride=1, name=name, op_surname='1')
        x = self._ConvNormRelu(x, num_channels=num_channels, kernel_size=3, stride=stride,
                               dilation=dilation, name=name, op_surname='2')
        x = self._Conv(x, num_channels=4 * num_channels, kernel_size=1, stride=1, name=name + '.conv3')
        x = batch_normalization_layer(x, is_training=self._is_training,
                                      momentum=self.parameters.batchNormalizationWeightDecay,
                                      epsilon=self.parameters.epsilon,
                                      use_scale=True, name=name + '.bn3')
        return x




    def _IdentityBlockBottleneck(self, tensor, num_channels, dilation=1, name=''):
        x = self._bottleneck_section(tensor, num_channels=num_channels, stride=1, dilation=dilation, name=name)

        x = x + tensor  # skip connection
        x = relu_layer(x, name=name + '.relu')

        return x

    def _IdentityBlockBasic(self, tensor, num_channels, dilation=1, name=''):
        x = self._basic_section(tensor, num_channels=num_channels, stride=1, dilation=dilation, name=name)

        x = x + tensor  # skip connection
        x = relu_layer(x, name=name + '.relu')

        return x


    def _ProjectionBlockBottleneck(self, tensor, num_channels, stride, dilation=1, name=''):
        if self.__is_emulating_PySOT:  #  For PyTorch-ResNet is "right_kernel = 1", for PySOT-ResNet is "right_kernel = 3"
            right_kernel = 3
        else:
            right_kernel = 1

        if int(tensor.shape[-1]) == num_channels:
            right_kernel = 1

        if dilation > 1:
            dilation = dilation // 2

        # left stream
        x = self._bottleneck_section(tensor, num_channels=num_channels, stride=stride, dilation=dilation, name=name)

        # right stream
        shortcut = self._Conv(tensor, num_channels=self.block_expansion * num_channels, kernel_size=right_kernel,
                              stride=stride, dilation=dilation, name=name + '.downsample.0')
        shortcut = batch_normalization_layer(shortcut, is_training=self._is_training,
                                             momentum=self.parameters.batchNormalizationWeightDecay,
                                             epsilon=self.parameters.epsilon, use_scale=True,
                                             name=name + '.downsample.1')

        x = shortcut + x  # skip connection
        x = relu_layer(x, name=name + '.relu')

        return x

    def _ProjectionBlockBasic(self, tensor, num_channels, stride, dilation=1, name=''):
        if self.__is_emulating_PySOT:  #  For PyTorch-ResNet is "right_kernel = 1", for PySOT-ResNet is "right_kernel = 3"
            right_kernel = 3
        else:
            right_kernel = 1
        dd = dilation

        if int(tensor.shape[-1]) == num_channels:
            downsample = False
        else:
            downsample = True

        if dilation > 1:
            dd = dilation // 2

        # left stream
        x = self._basic_section(tensor, num_channels=num_channels, stride=stride, dilation=dilation, name=name)


        # right stream
        if downsample:
            shortcut = self._Conv(tensor, num_channels=self.block_expansion * num_channels, kernel_size=right_kernel,
                                  stride=stride, dilation=dd, name=name + '.downsample.0')
            shortcut = batch_normalization_layer(shortcut, is_training=self._is_training,
                                                 momentum=self.parameters.batchNormalizationWeightDecay,
                                                 epsilon=self.parameters.epsilon, use_scale=True,
                                                 name=name + '.downsample.1')
        else:
            shortcut = tensor

        x = shortcut + x  # skip connection
        x = relu_layer(x, name=name + '.relu')

        return x


    def _ProjectionBlock(self, tensor, num_channels, stride, dilation=1, name=''):
        if self.block_type.lower() == 'Basic'.lower():
            x = self._ProjectionBlockBasic(tensor, num_channels=num_channels, stride=stride, dilation=dilation, name=name)

        elif self.block_type.lower() == 'Bottleneck'.lower():
            x = self._ProjectionBlockBottleneck(tensor, num_channels=num_channels, stride=stride, dilation=dilation, name=name)

        else:
            raise ValueError("Unrecognized block type {}".format(self.block_type))

        return x

    def _IdentityBlock(self, tensor, num_channels, dilation=1, name=''):
        if self.block_type.lower() == 'Basic'.lower():
            x = self._IdentityBlockBasic(tensor, num_channels=num_channels, dilation=dilation, name=name)

        elif self.block_type.lower() == 'Bottleneck'.lower():
            x = self._IdentityBlockBottleneck(tensor, num_channels=num_channels, dilation=dilation, name=name)

        else:
            raise ValueError("Unrecognized block type {}".format(self.block_type))

        return x

    def _Layer(self, x, num_channels, blocks, stride=1, dilation=1, name='layer'):
        x = self._ProjectionBlock(x, num_channels=num_channels, stride=stride, dilation=dilation, name=name + '.0')

        for i in range(1, blocks):
            x = self._IdentityBlock(x, num_channels=num_channels, dilation=dilation, name=name + '.{}'.format(i))

        return x


class SiameseBranch_ResNet18(SiameseBranch_ResNet):
    filter_size = np.array([15, 15])

    flavour = '18'

    def __init__(self, parameters):
        super(SiameseBranch_ResNet18, self).__init__(parameters=parameters)


class SiameseBranch_ResNet34(SiameseBranch_ResNet):
    filter_size = np.array([15, 15])

    flavour = '34'

    def __init__(self, parameters):
        super(SiameseBranch_ResNet34, self).__init__(parameters=parameters)


class SiameseBranch_ResNet50(SiameseBranch_ResNet):
    filter_size = np.array([15, 15])

    flavour = '50'

    def __init__(self, parameters):
        super(SiameseBranch_ResNet50, self).__init__(parameters=parameters)


class SiameseBranch_ResNet101(SiameseBranch_ResNet):
    filter_size = np.array([15, 15])

    flavour = '101'

    def __init__(self, parameters):
        super(SiameseBranch_ResNet101, self).__init__(parameters=parameters)


class SiameseBranch_ResNet152(SiameseBranch_ResNet):
    filter_size = np.array([15, 15])

    flavour = '152'

    def __init__(self, parameters):
        super(SiameseBranch_ResNet152, self).__init__(parameters=parameters)
