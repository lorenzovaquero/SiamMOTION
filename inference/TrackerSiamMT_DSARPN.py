from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from SiamMT.SiamMT_DSARPN import SiamMT_DSARPN
from inference.TrackerSiamMT_RPN import TrackerSiamMT_RPN


__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class TrackerSiamMT_DSARPN(TrackerSiamMT_RPN):
    """Allows an easy and homogeneous tracking, using an InferenceNetwork"""

    @property
    def _special_case_variables(self):
        new_dict = {}
        return new_dict

    @property
    def _exemplar_variables_dict_BASIC(self):
        new_dict = {}
        if self.parameters.network.Attention.useCrossAttention:
            new_dict.update({'exemplar_features': self.network_output_tensor_dict.get('exemplar_features')})
            new_dict.update({'exemplar_channel_attention_kernel': self.network_output_tensor_dict.get('exemplar_channel_attention_kernel')})

        if self.parameters.network.Attention.useSelfAttention:
            new_dict.update({'exemplar_self_attention_features': self.network_output_tensor_dict.get('exemplar_self_attention_features')})

        return new_dict


    def _get_inference_network(self):
        return SiamMT_DSARPN(parameters=self.parameters,
                             loader_as_feed_dict=self.loader_as_feed_dict,
                             cache_as_placeholder=False)


    def __init__(self, parameters, loader_as_feed_dict=True):
        super(TrackerSiamMT_DSARPN, self).__init__(parameters=parameters, loader_as_feed_dict=loader_as_feed_dict)


    def _assign_kernels(self):
        if hasattr(self.inference_network, 'placeholder_exemplar_features') and self.inference_network.placeholder_exemplar_features is not None:
            self.inference_network.placeholder_exemplar_features.cached_tensor.load(self.exemplar_features, self.session)

        if hasattr(self.inference_network, 'placeholder_exemplar_channel_attention_kernel') and self.inference_network.placeholder_exemplar_channel_attention_kernel is not None:
            self.inference_network.placeholder_exemplar_channel_attention_kernel.cached_tensor.load(self.exemplar_channel_attention_kernel, self.session)

        if hasattr(self.inference_network, 'placeholder_exemplar_self_attention_features') and self.inference_network.placeholder_exemplar_self_attention_features is not None:
            self.inference_network.placeholder_exemplar_self_attention_features.cached_tensor.load(self.exemplar_self_attention_features, self.session)


    @property
    def _cache_as_placeholder_kernel_dict(self):
        new_dict = {}
        if 'exemplar_features_placeholder' in self.network_input_tensor_dict:
            new_dict.update({self.network_input_tensor_dict['exemplar_features_placeholder']: self.exemplar_features})

        if 'exemplar_channel_attention_kernel_placeholder' in self.network_input_tensor_dict:
            new_dict.update({self.network_input_tensor_dict['exemplar_channel_attention_kernel_placeholder']: self.exemplar_channel_attention_kernel})

        if 'exemplar_self_attention_features_placeholder' in self.network_input_tensor_dict:
            new_dict.update({self.network_input_tensor_dict['exemplar_self_attention_features_placeholder']: self.exemplar_self_attention_features})
        return new_dict


    def _remove_variable_special_case(self, name, index):
        raise NotImplementedError("DSARPN shouldn't have special variables")


    def _update_variable_special_case(self, name, value_structure, index, new_value):
        raise NotImplementedError("DSARPN shouldn't have special variables")
