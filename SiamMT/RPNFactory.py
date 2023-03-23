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


class RPNFactory(object):

    @staticmethod
    def get_rpn(flavour='DepthwiseRPN'):

        if flavour.lower() == 'DepthwiseRPN'.lower():
            from neuralNetwork.RPNLayer_Depthwise import RPNLayer_Depthwise
            backbone = RPNLayer_Depthwise

        else:
            raise ValueError('RPN type "{}" isn\'t supported!'.format(flavour))

        return backbone
