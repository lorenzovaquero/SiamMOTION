from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from SiamMT.SimilarityOperationRPN import SimilarityOperationRPN
from SiamMT.SimilarityOperationDSA import SimilarityOperationDSA
from SiamMT.UpdateTargetRPN import UpdateTargetRPN
from SiamMT.CachePlaceholder import CachePlaceholder
from SiamMT.CacheVariable import CacheVariable
from SiamMT.SiamMT import SiamMT

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class SiamMT_DSARPN(SiamMT):

    is_training = False
    name = 'SiamMT-DSARPN'

    def __init__(self, parameters, loader_as_feed_dict=True, cache_as_placeholder=True):
        self.placeholder_exemplar_features = None
        self.placeholder_exemplar_self_attention_features = None
        self.placeholder_exemplar_channel_attention_kernel = None
        super(SiamMT_DSARPN, self).__init__(parameters=parameters, loader_as_feed_dict=loader_as_feed_dict,
                                            cache_as_placeholder=cache_as_placeholder)
        self.attention = SimilarityOperationDSA(self.parameters.network.Attention)


    def __choose_correct_cache(self):
        if self.cache_as_placeholder:
            return CachePlaceholder()
        else:
            return CacheVariable()

    def _get_cached_tensors(self):
        if self.parameters.network.Attention.useCrossAttention:
            self.placeholder_exemplar_features = self.__choose_correct_cache()
            self.placeholder_exemplar_channel_attention_kernel = self.__choose_correct_cache()

        if self.parameters.network.Attention.useSelfAttention:
            self.placeholder_exemplar_self_attention_features = self.__choose_correct_cache()


    def _get_similarity_operation(self):
        similarity = SimilarityOperationRPN(parameters=self.parameters.network.RPNLayer, is_training=self.is_training)
        return similarity

    def _get_update_target(self):
        update = UpdateTargetRPN(parameters=self.parameters.RPNInference, is_training=self.is_training)
        return update


    def __cache_tensor(self, input_tensor, cache_placeholder, name, output_tensor_dict, input_tensor_dict):
        output_tensor_dict[name] = input_tensor

        if self.cache_as_placeholder:
            cls_cache_name = name + '_placeholder'
        else:
            cls_cache_name = name + '_variable'

        output_tensor = cache_placeholder.build(input_tensor=input_tensor, name=cls_cache_name)

        input_tensor_dict[cls_cache_name] = output_tensor

        if not self.cache_as_placeholder:
            self.names_cached_as_variable.add(cache_placeholder.cached_tensor.name)

        return output_tensor


    def _build_similarity_and_update(self, exemplar_features, searcharea_features, input_tensor_dict, output_tensor_dict):
        exemplar_self_attention_features, \
        exemplar_channel_attention_kernel = self.attention.pre_build(exemplar_features_tensor=exemplar_features)

        if exemplar_channel_attention_kernel is not None:
            exemplar_features = self.__cache_tensor(
                input_tensor=exemplar_features,
                cache_placeholder=self.placeholder_exemplar_features,
                name='exemplar_features',
                output_tensor_dict=output_tensor_dict,
                input_tensor_dict=input_tensor_dict)

            exemplar_channel_attention_kernel = self.__cache_tensor(
                input_tensor=exemplar_channel_attention_kernel,
                cache_placeholder=self.placeholder_exemplar_channel_attention_kernel,
                name='exemplar_channel_attention_kernel',
                output_tensor_dict=output_tensor_dict,
                input_tensor_dict=input_tensor_dict)

        if exemplar_self_attention_features is not None:
            exemplar_self_attention_features = self.__cache_tensor(
                input_tensor=exemplar_self_attention_features,
                cache_placeholder=self.placeholder_exemplar_self_attention_features,
                name='exemplar_self_attention_features',
                output_tensor_dict=output_tensor_dict,
                input_tensor_dict=input_tensor_dict)


        exemplar_features, \
        searcharea_features = self.attention.build(exemplar_features_tensor=exemplar_features,
                                                   searcharea_features_tensor=searcharea_features,
                                                   exemplar_self_attention_features=exemplar_self_attention_features,
                                                   exemplar_channel_attention_kernel=exemplar_channel_attention_kernel)


        cls_exemplar_kernel, reg_exemplar_kernel = self.similarity.pre_build(exemplar_features_tensor=exemplar_features)


        cls_batch_tensor, reg_batch_tensor = self.similarity.build(
            cls_exemplar_kernel_tensor=cls_exemplar_kernel,
            reg_exemplar_kernel_tensor=reg_exemplar_kernel,
            searcharea_features_tensor=searcharea_features)

        update_pysot_factor = self.extractor.pysot_factor
        update_pysot_factor = tf.expand_dims(update_pysot_factor, axis=-1)

        updated_target_in_frame, updated_confidence = self.update.build(
            cls_batch_tensor=cls_batch_tensor,
            reg_batch_tensor=reg_batch_tensor,
            target_size_in_searcharea_tensor=self.extractor.target_size,
            pysot_factor=update_pysot_factor,
            searcharea_image_scale_factor_tensor=self.extractor.searcharea_image_scale_factor,
            searcharea_crop_target_in_frame_tensor=self.extractor.searcharea_crop_target_in_frame,
            frame_size_tensor=self.extractor.frame_size,
            virtual_searchAreaSize=self.extractor.virtual_searchAreaSize,
            prevent_out_of_bounds=True,
            min_target_size=10,
            frame_features_size_tensor=self.extractor.frame_features_size,
            # These are for extracting extra debug information
            backbone_filter_size=self.extractor.backbone.filter_size,
            backbone_stride=self.extractor.backbone.stride,
            searcharea_crop_target_in_frame_features_tensor=self.extractor.searcharea_crop_target_in_frame_features,
            searcharea_features_scale_factor_tensor=None,
            searcharea_features_size=self.extractor.searcharea_features_size
        )

        return updated_target_in_frame, updated_confidence