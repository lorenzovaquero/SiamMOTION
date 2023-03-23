"""InferenceNetwork.py: SiameseNetwork variety specifically intended for inference"""

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

from inference.inference_utils_tf import get_target_size
from SiamMT.SimilarityOperationRPN import SimilarityOperationRPN
from SiamMT.UpdateTargetRPN import UpdateTargetRPN
from SiamMT.CachePlaceholder import CachePlaceholder
from SiamMT.CacheVariable import CacheVariable
from SiamMT.SiamMT import SiamMT

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2018"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzovaquero@hotmail.com"
__status__ = "Prototype"


class SiamMT_RPN(SiamMT):
    """TODO"""

    is_training = False
    name = 'SiamMT-RPN'

    def __init__(self, parameters, loader_as_feed_dict=True, cache_as_placeholder=True):
        self.placeholder_cls = None
        self.placeholder_reg = None
        super(SiamMT_RPN, self).__init__(parameters=parameters, loader_as_feed_dict=loader_as_feed_dict,
                                         cache_as_placeholder=cache_as_placeholder)

    def _get_cached_tensors(self):
        if self.cache_as_placeholder:
            self.placeholder_cls = CachePlaceholder()
            self.placeholder_reg = CachePlaceholder()
        else:
            self.placeholder_cls = CacheVariable()
            self.placeholder_reg = CacheVariable()


    def _get_similarity_operation(self):
        similarity = SimilarityOperationRPN(parameters=self.parameters.network.RPNLayer, is_training=self.is_training)
        return similarity

    def _get_update_target(self):
        update = UpdateTargetRPN(parameters=self.parameters.RPNInference, is_training=self.is_training)
        return update

    def _build_similarity_and_update(self, exemplar_features, searcharea_features, input_tensor_dict, output_tensor_dict):
        cls_exemplar_kernel, reg_exemplar_kernel = self.similarity.pre_build(exemplar_features_tensor=exemplar_features)

        output_tensor_dict['cls_exemplar_kernel'] = cls_exemplar_kernel
        output_tensor_dict['reg_exemplar_kernel'] = reg_exemplar_kernel

        # Los nombramos distinto en funcion de si es placeholder o variable
        if self.cache_as_placeholder:
            cls_cache_name = 'cls_exemplar_kernel_placeholder'
            reg_cache_name = 'reg_exemplar_kernel_placeholder'
        else:
            cls_cache_name = 'cls_exemplar_kernel_variable'
            reg_cache_name = 'reg_exemplar_kernel_variable'

        cls_exemplar_kernel_placeholder = self.placeholder_cls.build(input_tensor=cls_exemplar_kernel,
                                                                     name=cls_cache_name)
        reg_exemplar_kernel_placeholder = self.placeholder_reg.build(input_tensor=reg_exemplar_kernel,
                                                                     name=reg_cache_name)
        input_tensor_dict[cls_cache_name] = cls_exemplar_kernel_placeholder
        input_tensor_dict[reg_cache_name] = reg_exemplar_kernel_placeholder

        if not self.cache_as_placeholder:
            self.names_cached_as_variable.add(self.placeholder_cls.cached_tensor.name)
            self.names_cached_as_variable.add(self.placeholder_reg.cached_tensor.name)

        cls_batch_tensor, reg_batch_tensor = self.similarity.build(
            cls_exemplar_kernel_tensor=cls_exemplar_kernel_placeholder,
            reg_exemplar_kernel_tensor=reg_exemplar_kernel_placeholder,
            searcharea_features_tensor=searcharea_features)

        # OJO!: Antes creia que el update_pysot_factor podia ser self.extractor.searcharea_features_scale_factor[..., 0]
        #  y que, de hecho, seria lo mismo que self.extractor.pysot_factor. Sin embargo, eso no es asi. Las diferencias
        #  entre ambos valores son "pequenas", pero, si no se reescala la imagen de entrada, con uno el sistema funciona
        #  MUY MAL, y con el otro el sistema funciona muy bien. No entiendo por que son diferentes ni por que solo
        #  funciona self.extractor.pysot_factor, pero es el que vamos a usar.
        update_pysot_factor = self.extractor.pysot_factor
        update_pysot_factor = tf.expand_dims(update_pysot_factor, axis=-1)

        updated_target_in_frame, updated_confidence = self.update.build(
            cls_batch_tensor=cls_batch_tensor,
            reg_batch_tensor=reg_batch_tensor,
            target_size_in_searcharea_tensor=self.extractor.target_size, #get_target_size(self.extractor.target_deviation_in_searcharea_image),
            pysot_factor=update_pysot_factor,
            searcharea_image_scale_factor_tensor=self.extractor.searcharea_image_scale_factor,
            searcharea_crop_target_in_frame_tensor=self.extractor.searcharea_crop_target_in_frame,
            frame_size_tensor=self.extractor.frame_size,
            virtual_searchAreaSize=self.extractor.virtual_searchAreaSize,
            prevent_out_of_bounds=True,
            min_target_size=10,
            frame_features_size_tensor=self.extractor.frame_features_size,
            # A partir de esta (incluida), son opcionales para sacar info extra
            backbone_filter_size=self.extractor.backbone.filter_size,
            backbone_stride=self.extractor.backbone.stride,
            searcharea_crop_target_in_frame_features_tensor=self.extractor.searcharea_crop_target_in_frame_features,
            searcharea_features_scale_factor_tensor=None,
            searcharea_features_size=self.extractor.searcharea_features_size
        )

        return updated_target_in_frame, updated_confidence