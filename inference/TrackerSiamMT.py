"""Tracker.py: Allows an easy and homogeneous tracking"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import logging
import abc

logging.basicConfig()
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from inference.inference_utils_tf import restore_inference_model

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class TrackerSiamMT(object):
    """Allows an easy and homogeneous tracking, using an InferenceNetwork"""

    __metaclass__ = abc.ABCMeta

    info_is_basic = False  # TODO: Set to True for testing

    _special_case_variables = {}

    @property
    def _exemplar_variables_dict_BASIC(self):
        return {
            'exemplar_kernel': self.network_output_tensor_dict.get('exemplar_kernel')
        }

    @property
    def _exemplar_variables_dict_FULL(self):
        new_dict = self._exemplar_variables_dict_BASIC.copy()
        new_dict.update({
            'exemplar_image': self.inference_network.extractor.exemplar_image,
            'exemplar_features': self.inference_network.extractor.exemplar_features,
            'exemplar_crop_target_in_frame': self.inference_network.extractor.exemplar_crop_target_in_frame,
            # 'exemplar_crop_target_in_frame_features': self.inference_network.extractor.exemplar_crop_target_in_frame_features,  # Lo comento porque con FPN no tiene demasiado sentido
            'current_searcharea_crop_target_in_frame': self.inference_network.extractor.searcharea_crop_target_in_frame,
            # 'current_searcharea_crop_target_in_frame_features': self.inference_network.extractor.searcharea_crop_target_in_frame_features  # Lo comento porque con FPN no tiene demasiado sentido
        })
        return new_dict

    @property
    def _exemplar_variables_dict(self):
        if self.info_is_basic:
            return self._exemplar_variables_dict_BASIC
        else:
            return self._exemplar_variables_dict_FULL

    @property
    def _exemplar_variables_names(self):
        return list(self._exemplar_variables_dict.keys())

    @property
    def exemplar_tensor_fetches(self):
        return list(self._exemplar_variables_dict.values())

    @property
    def exemplar_current_variables(self):
        return self.__property_getter(self._exemplar_variables_names)

    @exemplar_current_variables.setter
    def exemplar_current_variables(self, new_exemplar_variables):
        self.__property_setter(self._exemplar_variables_names, new_exemplar_variables)


    @property
    def _inference_variables_dict_BASIC(self):
        return {
            'current_target': self.network_output_tensor_dict.get('target'),
            'current_target_confidence': self.network_output_tensor_dict.get('target_confidence')
        }

    @property
    def _inference_variables_dict_FULL(self):
        new_dict = self._inference_variables_dict_BASIC.copy()
        new_dict.update({
            'current_target_preprocessed': self.inference_network.preprocessor.target_converted,
            'current_target_in_preprocessed_frame': self.inference_network.update.updated_target_in_frame,
            'current_target_deviation_in_searcharea_image': self.inference_network.update.updated_target_deviation_in_searcharea_image,

            # 'current_target_in_frame_features': self.inference_network.update.updated_target_in_frame_features,  # Lo comento porque con FPN no tiene demasiado sentido
            'current_target_in_searcharea_image': self.inference_network.update.updated_target_in_searcharea_image,
            # 'current_target_in_searcharea_features': self.inference_network.update.updated_target_in_searcharea_features,  # Lo comento porque con FPN no tiene demasiado sentido

            # 'previous_target_in_frame_features': self.inference_network.extractor.target_in_frame_features,  # Lo comento porque con FPN no tiene demasiado sentido
            'previous_target_deviation_in_searcharea_image': self.inference_network.extractor.target_deviation_in_searcharea_image,
            # 'previous_target_deviation_in_searcharea_features': self.inference_network.extractor.target_deviation_in_searcharea_features,  # Lo comento porque con FPN no tiene demasiado sentido

            'current_frame': self.inference_network.preprocessor.frame_tensor,
            'current_frame_preprocessed': self.inference_network.preprocessor.frame_converted,
            'current_frame_average_colors': self.inference_network.extractor.frame_average_colors,
            'current_frame_features': self.inference_network.extractor.frame_features,
            'current_searcharea_image': self.inference_network.extractor.searcharea_image,
            'current_searcharea_features': self.inference_network.extractor.searcharea_features,

            'current_frame_size_represented_inside_features': self.inference_network.extractor.frame_size_represented_inside_features,
            'current_searcharea_features_size_represented_inside_score': self.inference_network.similarity.searcharea_features_size_represented_inside_score,
            'current_frame_preprocessing_scale_factor': self.inference_network.preprocessor.frame_scale_factor_tensor,
            'current_frame_preprocessing_pad_topleft': self.inference_network.preprocessor.frame_pad_topleft_tensor,

            'current_searcharea_crop_target_in_frame': self.inference_network.extractor.searcharea_crop_target_in_frame,
            # 'current_searcharea_crop_target_in_frame_features': self.inference_network.extractor.searcharea_crop_target_in_frame_features,  # Lo comento porque con FPN no tiene demasiado sentido

            'current_scoremap': self.inference_network.similarity.scoremap,
            'current_penalized_scoremap': self.inference_network.update.penalized_scoremap_tensor,
            'penalization_window': self.inference_network.update.penalization_window,
            'global_penalization_window': self.inference_network.update.global_penalization_window,
        })
        return new_dict

    @property
    def _inference_variables_dict(self):
        if self.info_is_basic:
            return self._inference_variables_dict_BASIC
        else:
            return self._inference_variables_dict_FULL

    @property
    def _inference_variables_names(self):
        return list(self._inference_variables_dict.keys())

    @property
    def inference_tensor_fetches(self):
        return list(self._inference_variables_dict.values())

    @property
    def inference_current_variables(self):
        return self.__property_getter(self._inference_variables_names)

    @inference_current_variables.setter
    def inference_current_variables(self, new_inference_variables):
        self.__property_setter(self._inference_variables_names, new_inference_variables)


    def __property_getter(self, variables_names):
        return [getattr(self, v) for v in variables_names]

    def __property_setter(self, variables_names, new_variables_values):
        if new_variables_values is None:
            new_variables_values = [None] * len(variables_names)

        for i, variable_name in enumerate(variables_names):
            setattr(self, variable_name, new_variables_values[i])


    @abc.abstractmethod
    def _get_inference_network(self):
        raise NotImplementedError


    def __init__(self, parameters):
        self.parameters = parameters

        np.random.seed(self.parameters.randomSeed)
        tf.compat.v1.set_random_seed(self.parameters.randomSeed)

        self.session = None
        self.inference_network = self._get_inference_network()

        self.network_input_tensor_dict = {}
        self.network_output_tensor_dict = {}

        self.saver = None

        self.writer = None
        self.profiler = None
        self.logs_run_options = None
        self.logs_run_metadata = None

        self.frames_tracked = 0
        self.targets_added = 0

        # ====== vvv TRACKING INFO vvv ======
        self.current_frame = np.array([])

        self.current_target_identifier = np.array([])
        self.current_target = np.array([])
        self.current_target_confidence = np.array([])
        self.original_target_iou = np.array([])
        self.current_target_min_area = np.array([])
        self.current_target_max_area = np.array([])
        self.original_target = np.array([])

        for name in self._exemplar_variables_names:
            setattr(self, name, None)
        for name in self._inference_variables_names:
            setattr(self, name, None)
        # ====== ^^^ TRACKING INFO ^^^ ======


    def build(self, logs_folder=None, frame_path_list=None):
        """Initializes tracker's inference network, restoring it from a checkpoint model file
        frame_path_list is mandatory if we use a LoaderDataset"""

        logger.debug('Creating SiamMT-RPN Tracker')

        self.logs_folder = logs_folder
        self.frame_path_list = frame_path_list

        if self.logs_folder is not None:
            # We save the parameters for future references
            parameters_dict = self.parameters_to_dict(self.parameters)
            with open(os.path.join(self.logs_folder, 'parameters.json'), 'w') as fp:
                json.dump(parameters_dict, fp, indent=2)

            self.writer = tf.compat.v1.summary.FileWriter(self.logs_folder)

            self.logs_run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.logs_run_metadata = tf.compat.v1.RunMetadata()

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

        self.session = tf.compat.v1.Session(config=config)

        self.network_input_tensor_dict, self.network_output_tensor_dict = self.inference_network.build()

        if len(self.inference_network.trained_variables) > 0:
            self.saver = tf.compat.v1.train.Saver(var_list=self.inference_network.trained_variables)

            logger.debug('Restoring parameters from checkpoint {}'.format(self.parameters.modelFile))
            self._restore_from_checkpoint(model_file=self.parameters.modelFile)

        else:
            logger.warning('Model doesn\'t have any learned variables.')

        # We have to initialize the dataset if we use it
        if not self.inference_network.loader_as_feed_dict:
            logger.debug('Initialising dataset')
            self.session.run(self.network_input_tensor_dict['dataset_iterator'].initializer,
                             feed_dict={self.network_input_tensor_dict['dataset_frame_path_placeholder']: self.frame_path_list})

        if self.writer is not None:
            self.writer.add_graph(self.session.graph)
            self.writer.flush()

            self._profiler_timeline_path = os.path.join(self.logs_folder, 'ProfilerTimeline')
            self._profiler_operations_path = os.path.join(self.logs_folder, 'ProfilerOperations')
            os.mkdir(self._profiler_timeline_path)
            os.mkdir(self._profiler_operations_path)

            self.profiler = tf.compat.v1.profiler.Profiler(self.session.graph)

            # Profile the trainable parameters of the model.
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
                    .with_file_output(os.path.join(self.logs_folder, 'trainable_parameters.txt')).build())
            self.profiler.profile_name_scope(options=opts)

    @abc.abstractmethod
    def _rename_checkpoint_restoration_variables(self, variable_names_model, variable_names_checkpoint):
        raise NotImplementedError

    def _check_valid_checkpoint_rename_variables(self, new_variable_names, variable_names_model, variable_names_checkpoint):
        is_valid_rename = True
        extra_info = {}

        # "Momentum" isn't needed for inference
        any_momentum = any("/Momentum" in s for s in variable_names_model)
        if not any_momentum:
            variable_names_checkpoint = [val for val in variable_names_checkpoint if '/Momentum' not in val]

        # "Adam" isn't needed for inference
        any_adam = any("/Adam_" in s for s in variable_names_model)
        if not any_adam:
            variable_names_checkpoint = [val for val in variable_names_checkpoint if '/Adam_' not in val]

        # "step" isn't needed for inference
        variable_names_checkpoint = [val for val in variable_names_checkpoint if val != 'step']

        is_valid_rename &= set(new_variable_names.iterkeys()) == set(variable_names_checkpoint)
        is_valid_rename &= set(new_variable_names.itervalues()) == set(variable_names_model)

        if not is_valid_rename:
            extra_info['undefined_variables'] = list(set(variable_names_model) - set(new_variable_names.iterkeys()))
            extra_info['extra_variables'] = list(set(new_variable_names.iterkeys()) - set(variable_names_model))
        else:
            extra_info['undefined_variables'] = []
            extra_info['extra_variables'] = []

        return is_valid_rename, extra_info

    def _restore_from_checkpoint(self, model_file):
        if self.parameters.useRestoreForLoadingWeights:
            self.saver.restore(self.session, model_file)
        else:
            restore_inference_model(sess=self.session, ckpt_file=model_file, cached_variables=self.inference_network.names_cached_as_variable,
                                    use_old_method=self.parameters.useOldMethodForLoadingWeights)

    @abc.abstractmethod
    def _assign_kernels(self):
        raise NotImplementedError


    def _fix_grayscale(self, frame):
        if frame.shape[-1] == 1:
            logger.debug('Adjusting grayscale frame')
            color_frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
            frame = color_frame

        return frame

    def _update_numpy_value(self, old_value, added_value, axis=0):
        if old_value is None or old_value.size == 0:
            new_value = added_value
        else:
            new_value = np.append(old_value, added_value, axis=axis)

        return new_value


    def _add_exemplar_values(self, added_values):
        new_values = self.exemplar_current_variables

        for i, value in enumerate(added_values):
            if new_values[i] is None:
                new_values[i] = value
            else:
                axis = 0


                if self._exemplar_variables_names[i] in self._special_case_variables:
                    axis = 2 

                new_values[i] = self._update_numpy_value(new_values[i], value, axis=axis)

        self.exemplar_current_variables = new_values

    def _reset_exemplar_values(self, resetted_values, index):
        new_values = self.exemplar_current_variables

        for i, value in enumerate(resetted_values):


            if self._exemplar_variables_names[i] in self._special_case_variables:
                new_values[i] = self._update_variable_special_case(name=self._exemplar_variables_names[i],
                                                                   value_structure=new_values[i],
                                                                   index=index, new_value=value)
            else:
                new_values[i][index] = value

        self.exemplar_current_variables = new_values


    def __get_new_identifiers(self, num_identifiers):
        if self.current_target_identifier.size > 0:
            first_identifier = int(np.amax(self.current_target_identifier.astype("int32")) + 1)
        else:
            first_identifier = 0

        identifiers = list(range(first_identifier, first_identifier + num_identifiers))

        return identifiers

    def add_target(self, frame, target, identifier=None):
        """Processes a frame, saving the exemplar features of a new target.
        If identifier is not specified, an arbitrary unique identifier is assigned to the target
        "frame" must be None if we use a dataset
        "target" must have shape [4] or [None, 4] ([num_targets, [center_vertical, center_horizontal, height, width]])"""

        logger.debug('Adding target')
        self.targets_added += 1

        if frame is not None:
            frame = self._fix_grayscale(frame)

        if len(target.shape) == 1:
            target = np.expand_dims(target, axis=0)

        if identifier is None:
            identifier = self.__get_new_identifiers(num_identifiers=target.shape[0])

        self.current_target_identifier = np.append(self.current_target_identifier, identifier)

        self.current_target = self._update_numpy_value(self.current_target, target, axis=0)
        self.original_target = self._update_numpy_value(self.original_target, target, axis=0)
        self.current_target_confidence = self._update_numpy_value(self.current_target_confidence,
                                                                  np.array([1.0] * target.shape[0], dtype="float32"),
                                                                  axis=0)

        new_target_iou = np.array([0.0] * target.shape[0], dtype="float32")

        self.original_target_iou = self._update_numpy_value(self.original_target_iou,
                                                            new_target_iou,
                                                            axis=0)

        if self.parameters.sizeMinFactor > 0.0 and self.parameters.sizeMinFactor != 1.0:
            new_target_min_area = (target[:, 2] * target[:, 3]) * self.parameters.sizeMinFactor
            self.current_target_min_area = self._update_numpy_value(self.current_target_min_area, new_target_min_area, axis=0)

        if self.parameters.sizeMaxFactor > 0.0 and self.parameters.sizeMaxFactor != 1.0:
            new_target_max_area = (target[:, 2] * target[:, 3]) * self.parameters.sizeMaxFactor
            self.current_target_max_area = self._update_numpy_value(self.current_target_max_area, new_target_max_area, axis=0)

        if self.inference_network.loader_as_feed_dict:
            feed_dict = {self.network_input_tensor_dict['frame_placeholder']: frame,
                         self.network_input_tensor_dict['target_placeholder']: target}
        else:
            feed_dict = {self.network_input_tensor_dict['target_placeholder']: target}
            # We don't need a frame as we use a dataset

        if self.parameters.trackingLogsStep > 0 and self.targets_added % self.parameters.trackingLogsStep == 0:
            options = self.logs_run_options
            run_metadata = self.logs_run_metadata
        else:
            options = None
            run_metadata = None

        exemplar_current_values = self.session.run(self.exemplar_tensor_fetches,
                                                   feed_dict=feed_dict,
                                                   options=options,
                                                   run_metadata=run_metadata)

        if run_metadata is not None and self.writer is not None:
            self.writer.add_run_metadata(self.logs_run_metadata, "Add_step_{}".format(self.targets_added))

        self._add_exemplar_values(exemplar_current_values)

        if not self.inference_network.cache_as_placeholder:
            self._assign_kernels()

        return self.current_target

    @property
    @abc.abstractmethod
    def _cache_as_placeholder_kernel_dict(self):
        raise NotImplementedError

    def track_next_frame(self, frame):
        """Processes the next frame of a video, updating the targets"""
        logger.debug('Tracking next frame')
        self.frames_tracked += 1

        if frame is not None:
            frame = self._fix_grayscale(frame)

        self.current_frame = frame

        if self.current_target is None or self.current_target.size == 0: 
            return None, None 

        if self.inference_network.loader_as_feed_dict:
            feed_dict = {self.network_input_tensor_dict['frame_placeholder']: self.current_frame,
                         self.network_input_tensor_dict['target_placeholder']: self.current_target}
        else:
            feed_dict = {self.network_input_tensor_dict['target_placeholder']: self.current_target}
            # We don't need a frame as we use a dataset

        if self.inference_network.cache_as_placeholder:
            feed_dict.update(self._cache_as_placeholder_kernel_dict)
            # We don't need feed_dict for kernels if we use Variables

        if self.parameters.sizeMinFactor:
            feed_dict[self.network_input_tensor_dict['target_min_area_placeholder']] = self.current_target_min_area

        if self.parameters.sizeMaxFactor:
            feed_dict[self.network_input_tensor_dict['target_max_area_placeholder']] = self.current_target_max_area

        if self.parameters.trackingLogsStep > 0 and self.frames_tracked % self.parameters.trackingLogsStep == 0:
            options = self.logs_run_options
            run_metadata = self.logs_run_metadata
        else:
            options = None
            run_metadata = None

        self.inference_current_variables = self.session.run(self.inference_tensor_fetches,
                                                            feed_dict=feed_dict,
                                                            options=options,
                                                            run_metadata=run_metadata)

        if run_metadata is not None and self.writer is not None:
            self.writer.add_run_metadata(self.logs_run_metadata, "Track_step_{}".format(self.frames_tracked))

        if run_metadata is not None and self.profiler is not None:
            self.profiler.add_step(self.frames_tracked, run_metadata)

            # Profile the timing of the model operations.
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory())
                    .with_empty_output()
                    .with_node_names(hide_name_regexes=['.*/read', '.*/Assign', '.*/StopGradient', '.*/Regularizer/.*',
                                                        'ConstantFolding/.*', '.*/batch_normalization/moving_mean',
                                                        '.*/batch_normalization/moving_variance', '.*/misc/.*',
                                                        '.*/searcharea_and_exemplar_image_creation/.*',
                                                        '.*/conv/biases', '.*/conv/weights', '.*/exemplar_kernel_variable',
                                                        '.*/batch_normalization/beta', '.*/batch_normalization/gamma',
                                                        '.*/adjust/adjust_factor', '.*/adjust/bias',
                                                        '.*/clip_by_value(_\d)?/(x|y)', '.*/mul(_\d)?/(x|y)',
                                                        '.*/truediv(_\d)?/(x|y)', '.*/sub(_\d)?/(x|y)', '(.*/)?Const',
                                                        '.*/searcharea_and_exemplar_size_in_frame/.*'])
                    .account_displayed_op_only(True)
                    .with_step(self.frames_tracked)
                    .order_by('name')
                    .with_file_output(os.path.join(self._profiler_operations_path, 'operations_timing_{}'.format(self.frames_tracked))).build())
            self.profiler.profile_graph(options=opts)

            # Generate a timeline:
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory())
                    .with_step(self.frames_tracked)
                    .with_timeline_output(os.path.join(self._profiler_timeline_path, 'profiler_timeline')).build())
            self.profiler.profile_graph(options=opts)

        return self.current_target, self.current_target_confidence


    def _get_index_from_identifier(self, identifier=None):
        if identifier is None:
            index = 0

        elif self.current_target_identifier is not None and identifier in self.current_target_identifier:
            index = list(self.current_target_identifier).index(identifier)

        else:
            index = None

        return index

    @property
    def num_targets(self):
        return len(self.current_target_identifier)

    @abc.abstractmethod
    def _remove_variable_special_case(self, name, index):
        raise NotImplementedError

    @abc.abstractmethod
    def _update_variable_special_case(self, name, value_structure, index, new_value):
        raise NotImplementedError

    def remove_target(self, identifier=None):
        """Removes a target by its identifier. If identifier is not specified, the oldest target is removed"""
        logger.debug('Removing target')

        index = self._get_index_from_identifier(identifier=identifier)
        if index is None:
            raise KeyError('Target identifier "{}" not found'.format(identifier))

        if self.current_target is None or len(self.current_target) == 0:
            logger.warning("Target list is empty, couldn't remove target \"{}\" ({})".format(identifier, index))
            return

        for name in self._exemplar_variables_names:
            if name in self._special_case_variables:
                self._remove_variable_special_case(name=name, index=index)
            else:
                setattr(self, name, np.delete(getattr(self, name), index, 0))

        self.current_target_identifier = np.delete(self.current_target_identifier, index, 0)
        self.original_target = np.delete(self.original_target, index, 0)
        self.current_target = np.delete(self.current_target, index, 0)
        self.current_target_confidence = np.delete(self.current_target_confidence, index, 0)
        self.original_target_iou = np.delete(self.original_target_iou, index, 0)

        if self.parameters.sizeMinFactor > 0.0 and self.parameters.sizeMinFactor != 1.0:
            self.current_target_min_area = np.delete(self.current_target_min_area, index, 0)

        if self.parameters.sizeMaxFactor > 0.0 and self.parameters.sizeMaxFactor != 1.0:
            self.current_target_max_area = np.delete(self.current_target_max_area, index, 0)

        if not self.inference_network.cache_as_placeholder:
            self._assign_kernels()


    def reset_target(self, frame, target, identifier, reset_visual_information=True):
        """Resets a target, updating its bounding-box and exemplar."""
        logger.debug('Resetting target')

        index = self._get_index_from_identifier(identifier=identifier)
        if index is None:
            raise KeyError('Target identifier "{}" not found'.format(identifier))

        if self.current_target is None or len(self.current_target) == 0:
            logger.warning("Target list is empty, couldn't remove target \"{}\" ({})".format(identifier, index))
            return

        if frame is not None:
            frame = self._fix_grayscale(frame)

        if len(target.shape) == 1:
            target = np.expand_dims(target, axis=0)

        self.current_target[index] = target[0]
        self.original_target[index] = target[0]
        self.current_target_confidence[index] = 1.0

        if self.parameters.sizeMinFactor > 0.0 and self.parameters.sizeMinFactor != 1.0:
            new_target_min_area = (target[:, 2] * target[:, 3]) * self.parameters.sizeMinFactor
            self.current_target_min_area[index] = new_target_min_area

        if self.parameters.sizeMaxFactor > 0.0 and self.parameters.sizeMaxFactor != 1.0:
            new_target_max_area = (target[:, 2] * target[:, 3]) * self.parameters.sizeMaxFactor
            self.current_target_max_area[index] = new_target_max_area

        if not reset_visual_information:  # If false, overrides updateFeaturesByIoU
            print('No actualizo features para {}'.format(identifier))
            return self.current_target

        if self.inference_network.loader_as_feed_dict:
            feed_dict = {self.network_input_tensor_dict['frame_placeholder']: frame,
                         self.network_input_tensor_dict['target_placeholder']: target}
        else:
            feed_dict = {self.network_input_tensor_dict['target_placeholder']: target}
            # We don't need a frame as we use a dataset

        if self.parameters.trackingLogsStep > 0 and self.targets_added % self.parameters.trackingLogsStep == 0:
            options = self.logs_run_options
            run_metadata = self.logs_run_metadata
        else:
            options = None
            run_metadata = None

        exemplar_current_values = self.session.run(self.exemplar_tensor_fetches,
                                                   feed_dict=feed_dict,
                                                   options=options,
                                                   run_metadata=run_metadata)

        if run_metadata is not None and self.writer is not None:
            self.writer.add_run_metadata(self.logs_run_metadata, "Add_step_{}".format(self.targets_added))

        self._reset_exemplar_values(exemplar_current_values, index=index)

        if not self.inference_network.cache_as_placeholder:
            self._assign_kernels()

        return self.current_target


    def close(self):
        """The tracker is destroyed, freeing its resources"""

        logger.debug('Closing tracker')

        if self.writer is not None:
            self.writer.flush()

        if self.profiler is not None:
            ALL_ADVICE = {'ExpensiveOperationChecker': {},
                          'AcceleratorUtilizationChecker': {},
                          'JobChecker': {},  # Only available internally.
                          'OperationChecker': {}}
            self.profiler.advise(ALL_ADVICE)


        if self.logs_run_metadata is not None:
            tl = timeline.Timeline(self.logs_run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(os.path.join(self.logs_folder, 'timeline.json'), 'w') as f:
                f.write(ctf)

        if self.session is not None:
            self.session.close()

    @staticmethod
    def parameters_to_dict(parameters):
        parameters_type = type(parameters)

        parameters_dict = copy.deepcopy(vars(parameters))
        for par_key, par_value in parameters_dict.iteritems():
            if type(par_value) == parameters_type:
                parameters_dict[par_key] = TrackerSiamMT.parameters_to_dict(par_value)

            elif type(par_value) == np.ndarray:
                parameters_dict[par_key] = np.copy(par_value).tolist()

        return parameters_dict