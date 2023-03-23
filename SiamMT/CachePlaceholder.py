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


class CachePlaceholder(object):

    def __init__(self):
        self.cached_tensor = None


    def build(self, input_tensor, name='cached_tensor'):
        # It has to be done this way in order to 'break the flow' and feed always the same exemplar_kernels to the model
        placeholder_dimensions = []

        for shape in input_tensor.get_shape():
            try:
                shape_val = int(shape)
            except:
                shape_val = None

            placeholder_dimensions.append(shape_val)

        self.cached_tensor = tf.compat.v1.placeholder(input_tensor.dtype, placeholder_dimensions, name=name)

        return self.cached_tensor
