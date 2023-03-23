from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
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


class BackboneFactory(object):

    @staticmethod
    def get_backbone(flavour='ResNet-18'):
        if flavour.lower().startswith('ResNet'.lower()):
            resnet_flavour = flavour.lower()[len('ResNet'):]

            if resnet_flavour == '18':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet18
                backbone = SiameseBranch_ResNet18

            elif resnet_flavour == '34':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet34
                backbone = SiameseBranch_ResNet34

            elif resnet_flavour == '50':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet50
                backbone = SiameseBranch_ResNet50

            elif resnet_flavour == '101':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet101
                backbone = SiameseBranch_ResNet101

            elif resnet_flavour == '152':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet152
                backbone = SiameseBranch_ResNet152

            else:
                raise ValueError('ResNet type "{}" isn\'t supported!'.format(resnet_flavour))

        else:
            raise ValueError('Backbone type "{}" isn\'t supported!'.format(flavour))

        return backbone
