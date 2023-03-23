"""Tracker.py: Allows an easy and homogeneous tracking"""

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

from SiamMT.SiamMT_RPN import SiamMT_RPN
from inference.TrackerSiamMT import TrackerSiamMT


__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class TrackerSiamMT_RPN(TrackerSiamMT):
    """Allows an easy and homogeneous tracking, using an InferenceNetwork"""

    _special_case_variables = {'cls_exemplar_kernel', 'reg_exemplar_kernel'}

    @property
    def _exemplar_variables_dict_BASIC(self):
        return {
            'cls_exemplar_kernel': self.network_output_tensor_dict.get('cls_exemplar_kernel'),
            'reg_exemplar_kernel': self.network_output_tensor_dict.get('reg_exemplar_kernel')
        }

    @property
    def _inference_variables_dict_FULL(self):
        new_dict = super(TrackerSiamMT_RPN, self)._inference_variables_dict_FULL.copy()
        new_dict.update({
            'current_cls_batch': self.inference_network.similarity.cls_batch_tensor,
            'current_reg_batch': self.inference_network.similarity.reg_batch_tensor
        })
        return new_dict


    def _get_inference_network(self):
        return SiamMT_RPN(parameters=self.parameters,
                          loader_as_feed_dict=self.loader_as_feed_dict,
                          cache_as_placeholder=False)


    def __init__(self, parameters, loader_as_feed_dict=True):
        self.loader_as_feed_dict = loader_as_feed_dict
        super(TrackerSiamMT_RPN, self).__init__(parameters=parameters)


    def _rename_checkpoint_restoration_variables(self, variable_names_model, variable_names_checkpoint):
        # We create a 'translation' dict using the checkpoint var list and a copy
        new_variable_names = dict(zip(variable_names_checkpoint, list(variable_names_checkpoint)))

        for key, value in new_variable_names.iteritems():
            if 'siamese/' in value:
                new_variable_names[key] = value.replace('siamese', 'SiamMT/FeatureExtractor/frame_features/siamese', 1)

            elif 'RPN/' in value:
                if 'conv_exemplar/' in value:
                    new_variable_names[key] = value.replace('RPN', 'SiamMT/SimilarityOperation/kernels_creation/RPN', 1)
                else:
                    new_variable_names[key] = value.replace('RPN', 'SiamMT/SimilarityOperation/xcorr/RPN', 1)

        # Maybe the previous model didn't have momentum
        any_momentum = any("/Momentum" in s for s in variable_names_model)
        for key in list(new_variable_names.keys()):
            if '/Momentum' in key and not any_momentum:
                del new_variable_names[key]

        # Maybe the previous model didn't have Adam
        any_adam = any("/Adam" in s for s in variable_names_model)
        for key in list(new_variable_names.keys()):
            if '/Adam' in key and not any_adam:
                del new_variable_names[key]

        # Maybe the previous model didn't have TrainOperation
        any_train = any("/TrainOperation/" in s for s in variable_names_model)
        for key in list(new_variable_names.keys()):
            if '/TrainOperation/' in key and not any_train:
                del new_variable_names[key]

        # In inference we won't have "step"
        if 'step' in list(new_variable_names.keys()) and 'step' not in variable_names_model:
            del new_variable_names['step']

        return new_variable_names


    def _assign_kernels(self):
        self.inference_network.placeholder_cls.cached_tensor.load(self.cls_exemplar_kernel, self.session)
        self.inference_network.placeholder_reg.cached_tensor.load(self.reg_exemplar_kernel, self.session)


    @property
    def _cache_as_placeholder_kernel_dict(self):
        return {self.network_input_tensor_dict['cls_exemplar_kernel_placeholder']: self.cls_exemplar_kernel,
                self.network_input_tensor_dict['reg_exemplar_kernel_placeholder']: self.reg_exemplar_kernel}


    def _remove_variable_special_case(self, name, index):
        kernel_num_channels = self.cls_exemplar_kernel.shape[2] // self.num_targets

        if name in {'cls_exemplar_kernel', 'reg_exemplar_kernel'}:
            setattr(self, name, np.delete(getattr(self, name),
                                          [kernel_num_channels * index + s for s in range(kernel_num_channels)],
                                          axis=2)
                    )

        else:
            logger.warning('Remove special case "{}" not configured'.format(name))


    def _update_variable_special_case(self, name, value_structure, index, new_value):
        kernel_num_channels = self.cls_exemplar_kernel.shape[2] // self.num_targets

        if name in {'cls_exemplar_kernel', 'reg_exemplar_kernel'}:
            value_structure[:, :, kernel_num_channels * index : kernel_num_channels * index + kernel_num_channels] = new_value

        else:
            logger.warning('Remove special case "{}" not configured'.format(name))

        return value_structure