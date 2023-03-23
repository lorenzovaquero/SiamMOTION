from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import logging
import abc

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from SiamMT.LoaderFeedDict import LoaderFeedDict
from SiamMT.LoaderDataset import LoaderDataset
from SiamMT.FeatureExtractor import FeatureExtractor
from SiamMT.Preprocessor import Preprocessor
from SiamMT.Postprocessor import Postprocessor

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class SiamMT(object):

    __metaclass__ = abc.ABCMeta

    is_training = False
    name = None  # Name of the SiamMT network (child classes must override this parameter)

    def __get_loader(self):
        if self.loader_as_feed_dict:
            loader = LoaderFeedDict()
        else:
            loader = LoaderDataset()

        return loader

    @abc.abstractmethod
    def _get_cached_tensors(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_similarity_operation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_update_target(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_similarity_and_update(self, exemplar_features, searcharea_features, input_tensor_dict, output_tensor_dict):
        raise NotImplementedError

    def __init__(self, parameters, loader_as_feed_dict=True, cache_as_placeholder=True):
        self.parameters = parameters
        self.loader_as_feed_dict = loader_as_feed_dict
        self.cache_as_placeholder = cache_as_placeholder

        self.names_cached_as_variable = set()

        self.loader = self.__get_loader()

        self.preprocessor = Preprocessor(parameters=self.parameters.Preprocessor)
        self.extractor = FeatureExtractor(parameters=self.parameters.network.backbone, is_training=self.is_training)
        self.similarity = self._get_similarity_operation()
        self.update = self._get_update_target()
        self.postprocessor = Postprocessor()



        self._get_cached_tensors()

        self.scope = None
        self.trained_variables = None


    def build(self, name='SiamMT'):
        """Builds an inference network"""

        logger.debug('Creating SiamMT ({})'.format(self.name))

        with tf.compat.v1.variable_scope(name) as self.scope:
            input_tensor_dict = {}
            output_tensor_dict = {}

            if self.loader_as_feed_dict:
                input_frame_tensor, input_target_tensor, input_frame_placeholder, input_target_placeholder = self.loader.build(load_as_opencv=self.parameters.loadAsOpenCV)
                input_tensor_dict['frame_placeholder'] = input_frame_placeholder
                input_tensor_dict['target_placeholder'] = input_target_placeholder

            else:
                input_frame_tensor, input_target_tensor, dataset_frame_path_placeholder, dataset_iterator = self.loader.build(load_as_opencv=self.parameters.loadAsOpenCV)
                input_tensor_dict['dataset_iterator'] = dataset_iterator
                input_tensor_dict['dataset_frame_path_placeholder'] = dataset_frame_path_placeholder
                input_tensor_dict['target_placeholder'] = input_target_tensor

            frame_converted, target_converted = self.preprocessor.build(frame_tensor=input_frame_tensor,
                                                                        target_tensor=input_target_tensor,
                                                                        backbone_filter_size=self.extractor.backbone.filter_size,
                                                                        backbone_stride=self.extractor.backbone.stride,
                                                                        reference_exemplar_size=self.parameters.virtualExemplarSize,
                                                                        reference_searcharea_size=self.parameters.virtualSearchAreaSize)

            if hasattr(self.update.parameters, 'scaleFactors'):
                extractor_scale_factors = self.update.parameters.scaleFactors
            else:
                extractor_scale_factors = None
            exemplar_features, searcharea_features = self.extractor.build(frame_tensor=frame_converted,
                                                                          target_tensor=target_converted,
                                                                          scale_factors=extractor_scale_factors)

            updated_target_in_frame, updated_confidence = self._build_similarity_and_update(
                exemplar_features=exemplar_features,
                searcharea_features=searcharea_features,
                input_tensor_dict=input_tensor_dict,
                output_tensor_dict=output_tensor_dict)

            if self.parameters.sizeMinFactor > 0.0 and self.parameters.sizeMinFactor != 1.0:
                target_min_area_tensor = tf.compat.v1.placeholder(tf.float32, [None], name='target_min_area_tensor')
                input_tensor_dict['target_min_area_placeholder'] = target_min_area_tensor
            else:
                target_min_area_tensor = None

            if self.parameters.sizeMaxFactor > 0.0 and self.parameters.sizeMaxFactor != 1.0:
                target_max_area_tensor = tf.compat.v1.placeholder(tf.float32, [None], name='target_max_area_tensor')
                input_tensor_dict['target_max_area_placeholder'] = target_max_area_tensor
            else:
                target_max_area_tensor = None

            output_updated_target_in_frame = self.postprocessor.build(target_tensor=updated_target_in_frame,
                                                                      frame_pad_topleft_tensor=self.preprocessor.frame_pad_topleft_tensor,
                                                                      frame_scale_factor_tensor=self.preprocessor.frame_scale_factor_tensor,
                                                                      target_min_area=target_min_area_tensor,
                                                                      target_max_area=target_max_area_tensor)

            output_tensor_dict['target'] = output_updated_target_in_frame
            output_tensor_dict['target_confidence'] = updated_confidence

            self._get_trained_variables()

        return input_tensor_dict, output_tensor_dict


    def _get_trained_variables(self):
        all_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)


        if not self.cache_as_placeholder:
            learned_variables = [v for v in all_variables if v.name not in self.names_cached_as_variable]

        else:
            learned_variables = all_variables

        self.trained_variables = learned_variables
