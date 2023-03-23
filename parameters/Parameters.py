"""ParametersGeneral.py: Tuple containing Feature Extractor parameters"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import numpy as np
import logging
import copy
from argparse import Namespace

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from inference.inference_utils_tf import get_tensor_size_after_convolution, get_tensor_size_before_convolution

logger = logging.getLogger(__name__)

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class Parameters(Namespace):
    """Configuration for the SiamMT application"""

    @staticmethod
    def __default_dict():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        parameter_dict = {}

        # GENERAL
        parameter_dict['gpuId'] = -1
        parameter_dict['randomSeed'] = 1
        parameter_dict['logs_folder'] = os.path.join(root_path, 'logs')

        # RPN
        parameter_dict['RPN'] = {}
        parameters_rpn = parameter_dict['RPN']
        parameters_rpn['anchorSize'] = 8
        parameters_rpn['anchorScales'] = [8]
        parameters_rpn['anchorRatios'] = [0.33, 0.5, 1, 2, 3]

        # TRACKER
        parameter_dict['Tracker'] = {}
        parameters_tracker = parameter_dict['Tracker']
        parameters_tracker['similarity'] = "DSARPN"
        parameters_tracker['exemplarSize'] = [15, 15]
        parameters_tracker['searchAreaSize'] = [31, 31]
        parameters_tracker['contextAmount'] = 0.5  # -0.122047244094
        parameters_tracker['loadAsOpenCV'] = True
        parameters_tracker['updateFeaturesByIoU'] = False
        parameters_tracker['cropExemplarFeatures'] = 7
        parameters_tracker['modelPath'] = os.path.join(root_path, 'model', 'SiamMOTION')
        parameters_tracker['modelName'] = 'model_tf.ckpt'
        parameters_tracker['useRestoreForLoadingWeights'] = True
        parameters_tracker['useOldMethodForLoadingWeights'] = False
        parameters_tracker['trackingLogsFolderName'] = 'tracking'
        parameters_tracker['trackingLogsStep'] = 0
        parameters_tracker['sizeMinFactor'] = 0.2
        parameters_tracker['sizeMaxFactor'] = 5.0
        parameters_tracker['inertiaRT'] = False

        parameters_tracker['FCInference'] = {}
        parameters_tracker_FC = parameters_tracker['FCInference']
        parameters_tracker_FC['numScales'] = 3
        parameters_tracker_FC['scaleStep'] = 1.0375
        parameters_tracker_FC['scalePenalty'] = 0.9745
        parameters_tracker_FC['scaleDamping'] = 0.59
        parameters_tracker_FC['upsampleFactor'] = 16
        parameters_tracker_FC['upsampleMethod'] = 'bicubic'
        parameters_tracker_FC['windowing'] = 'cosine'
        parameters_tracker_FC['windowInfluence'] = 0.30

        parameters_tracker['RPNInference'] = {}
        parameters_tracker_RPN = parameters_tracker['RPNInference']
        parameters_tracker_RPN['scaleRatioPenalty'] = 0.05
        parameters_tracker_RPN['scaleDamping'] = 0.18
        parameters_tracker_RPN['windowing'] = 'cosine'
        parameters_tracker_RPN['windowInfluence'] = 0.42
        parameters_tracker_RPN['globalWindowSize'] = 0.10
        parameters_tracker_RPN['globalWindowInfluence'] = 0.01
        parameters_tracker_RPN['morphScoreFlavour'] = 'EROSION+DILATION'
        parameters_tracker_RPN['lrAsPysot'] = True
        parameters_tracker_RPN['bboxRefinement'] = 'VOTING'
        parameters_tracker_RPN['bboxRefVotingIoU'] = 0.8
        parameters_tracker_RPN['bboxRefVotingUsePscore'] = True
        parameters_tracker_RPN['bboxRefVotingUpdateScore'] = True
        parameters_tracker_RPN['bboxRefTopNum'] = 0
        parameters_tracker_RPN['bboxRefTopUseWeight'] = False
        parameters_tracker_RPN['smoothScoreWithLocationMean'] = False

        parameters_tracker['Preprocessor'] = {}
        parameters_tracker_RPN = parameters_tracker['Preprocessor']
        parameters_tracker_RPN['setFrameSize'] = None
        parameters_tracker_RPN['setExactSize'] = False
        parameters_tracker_RPN['minimumFrameSize'] = None
        parameters_tracker_RPN['frameSizeAccordingToSearchArea'] = False
        parameters_tracker_RPN['objectiveSearchAreaSize'] = 287
        parameters_tracker_RPN['objectiveSearchAreaPolicy'] = 'greater'
        parameters_tracker_RPN['objectiveSearchAreaMargin'] = 64
        parameters_tracker_RPN['invertRGBOrder'] = False
        parameters_tracker_RPN['keepPixelValues'] = True
        parameters_tracker_RPN['zeroCenterPixelValues'] = False
        parameters_tracker_RPN['padFrameEffectiveSize'] = False
        parameters_tracker_RPN['adjustFrameValidConvolution'] = False

        # TRAINER
        parameter_dict['Trainer'] = {}
        parameters_trainer = parameter_dict['Trainer']
        parameters_trainer['exemplarSize'] = [15, 15]
        parameters_trainer['searchAreaSize'] = [31, 31]
        parameters_trainer['contextAmount'] = 0.50  # -0.122047244094
        parameters_trainer['cropExemplarFeatures'] = 0
        parameters_trainer['loadAsOpenCV'] = True
        parameters_trainer['loadAsSiamTF'] = False  # Loads the images directly as search areas and exemplars
        parameters_trainer['loadForFPN'] = True
        parameters_trainer['loadForFPNHasEqualAug'] = True
        parameters_trainer['oobTargetsIouLimit'] = 0.75
        parameters_trainer['frameSizeLimit'] = [608, 1080]
        parameters_trainer['pretrainedModelFile'] = os.path.join(root_path, 'model', 'Pretrained')
        parameters_trainer['trainingLogsFolderName'] = 'training'
        parameters_trainer['trainingLogsStep'] = 50
        parameters_trainer['validationLogsStep'] = 1
        parameters_trainer['validateOverSameEpoch'] = True
        parameters_trainer['imageLogsStep'] = 150
        parameters_trainer['trainNumEpochs'] = 100
        parameters_trainer['frozenUntilEpoch'] = 100
        parameters_trainer['numPairs'] = 100000
        parameters_trainer['trainBatchSize'] = 24
        parameters_trainer['maxFrameRange'] = 100
        parameters_trainer['imdbTrainingRatio'] = 0.85
        parameters_trainer['trainNegativePairProbability'] = 0.3
        parameters_trainer['validationNegativePairProbability'] = 0.0
        parameters_trainer['momentum'] = 0.9
        parameters_trainer['clipGradientNorm'] = 0
        parameters_trainer['exemplarMaxScaleShift'] = 0.05
        parameters_trainer['exemplarMaxTranslationShift'] = 4
        parameters_trainer['searchAreaMaxScaleShift'] = 0.18
        parameters_trainer['searchAreaMaxTranslationShift'] = 64

        parameters_trainer['augmentation'] = {}
        parameters_trainer_augmentation = parameters_trainer['augmentation']

        parameters_trainer_augmentation['exemplar'] = {}
        parameters_trainer_augmentation_exemplar = parameters_trainer_augmentation['exemplar']
        parameters_trainer_augmentation_exemplar['maxScale'] = 0.05
        parameters_trainer_augmentation_exemplar['maxTranslation'] = 4
        parameters_trainer_augmentation_exemplar['maxRotation'] = 0.01
        parameters_trainer_augmentation_exemplar['maxColorFactor'] = 1.0
        parameters_trainer_augmentation_exemplar['maxBlurProbability'] = 0.0
        parameters_trainer_augmentation_exemplar['maxNoiseFactor'] = 0.01
        parameters_trainer_augmentation_exemplar['noiseStd'] = 0.01

        parameters_trainer_augmentation['searchArea'] = {}
        parameters_trainer_augmentation_searcharea = parameters_trainer_augmentation['searchArea']
        parameters_trainer_augmentation_searcharea['maxScale'] = 0.18
        parameters_trainer_augmentation_searcharea['maxTranslation'] = 64
        parameters_trainer_augmentation_searcharea['maxRotation'] = 0.01
        parameters_trainer_augmentation_searcharea['maxColorFactor'] = 1.0
        parameters_trainer_augmentation_searcharea['maxBlurProbability'] = 0.2
        parameters_trainer_augmentation_searcharea['maxNoiseFactor'] = 0.02
        parameters_trainer_augmentation_searcharea['noiseStd'] = 0.05

        parameters_trainer['learningRate'] = {}
        parameters_trainer_learning = parameters_trainer['learningRate']
        parameters_trainer_learning['trainLearningRateStart'] = 3e-5
        parameters_trainer_learning['trainLearningRateStop'] = 3e-5
        parameters_trainer_learning['trainLearningRatePolicy'] = 'exponential'

        parameters_trainer['FCTraining'] = {}
        parameters_trainer_FC = parameters_trainer['FCTraining']
        parameters_trainer_FC['positiveRadius'] = 16
        parameters_trainer_FC['neutralRadius'] = 86
        parameters_trainer_FC['isZeroNegative'] = True
        parameters_trainer_FC['lossType'] = "SiamTF"

        parameters_trainer['RPNTraining'] = {}
        parameters_trainer_RPN = parameters_trainer['RPNTraining']
        parameters_trainer_RPN['maxPositiveAnchors'] = 16
        parameters_trainer_RPN['samplesWhenNegativeExample'] = 32
        parameters_trainer_RPN['maxContributingAnchors'] = 64
        parameters_trainer_RPN['iouNegativeThreshold'] = 0.3
        parameters_trainer_RPN['iouPositiveThreshold'] = 0.6
        parameters_trainer_RPN['maxIouAsPositive'] = True
        parameters_trainer_RPN['lossClsDelta'] = 1.0
        parameters_trainer_RPN['lossBalanceClsReg'] = 1.2
        parameters_trainer_RPN['weightDecay'] = 5e-6

        # NETWORK
        parameter_dict['network'] = {}
        parameters_network = parameter_dict['network']

        parameters_network['backbone'] = {}
        parameters_network_backbone = parameters_network['backbone']
        parameters_network_backbone['branchType'] = 'ResNet18'
        parameters_network_backbone['useDilation'] = False
        parameters_network_backbone['usedLayers'] = [1, 2, 3, 4]
        parameters_network_backbone['padding'] = 'SAME'
        parameters_network_backbone['adjust'] = "FPN"
        parameters_network_backbone['standardizeInputImage'] = True
        parameters_network_backbone['perImageStandardization'] = False
        parameters_network_backbone['grouped'] = False
        parameters_network_backbone['convolutionWeightDecay'] = 5e-6
        parameters_network_backbone['convolutionBiasDecay'] = 0.0
        parameters_network_backbone['batchNormalizationWeightDecay'] = 0.95
        parameters_network_backbone['initMethod'] = 'kaiming'
        parameters_network_backbone['stddev'] = 0.01
        parameters_network_backbone['epsilon'] = 1e-5
        parameters_network_backbone['cropSizeStyle'] = 'siamrpn'
        parameters_network_backbone['tensorResizeMethod'] = 'bilinear'

        parameters_network['FCLayer'] = {}
        parameters_network_FC = parameters_network['FCLayer']
        parameters_network_FC['adjustFactor'] = 0.001
        parameters_network_FC['singleSearchArea'] = False

        parameters_network['RPNLayer'] = {}
        parameters_network_RPN = parameters_network['RPNLayer']
        parameters_network_RPN['rpnType'] = "DepthwiseRPN"
        parameters_network_RPN['adjustFactor'] = 1.0
        parameters_network_RPN['convolutionWeightDecay'] = 5e-6
        parameters_network_RPN['batchNormalizationWeightDecay'] = 0.95
        parameters_network_RPN['epsilon'] = 1e-5
        parameters_network_RPN['initMethod'] = 'kaiming'
        parameters_network_RPN['flipRegressionCoordinates'] = False 

        parameters_network['Attention'] = {}
        parameters_network_Attention = parameters_network['Attention']
        parameters_network_Attention['useSelfAttention'] = True
        parameters_network_Attention['useCrossAttention'] = True
        parameters_network_Attention['ignoreSpatialAttention'] = True
        parameters_network_Attention['convolutionWeightDecay'] = 5e-6


        # CURATION
        parameter_dict['curation'] = {}
        parameters_curation = parameter_dict['curation']
        parameters_curation['exemplarSize'] = [127, 127]
        parameters_curation['searchAreaSize'] = [271, 271]
        parameters_curation['contextAmount'] = 0.5
        parameters_curation['searchAreaAugmentationMargin'] = 8

        return parameter_dict

    @staticmethod
    def __get_precalculated(parameter_dict):


        parameter_dict['RPN']['anchorScales'] = np.array(parameter_dict['RPN']['anchorScales'], dtype='float32')
        parameter_dict['RPN']['anchorRatios'] = np.array(parameter_dict['RPN']['anchorRatios'], dtype='float32')

        parameter_dict['Tracker']['exemplarSize'] = np.array(parameter_dict['Tracker']['exemplarSize'], dtype='int32')
        parameter_dict['Tracker']['searchAreaSize'] = np.array(parameter_dict['Tracker']['searchAreaSize'], dtype='int32')

        parameter_dict['Trainer']['exemplarSize'] = np.array(parameter_dict['Trainer']['exemplarSize'], dtype='int32')
        parameter_dict['Trainer']['searchAreaSize'] = np.array(parameter_dict['Trainer']['searchAreaSize'], dtype='int32')

        parameter_dict['curation']['exemplarSize'] = np.array(parameter_dict['curation']['exemplarSize'], dtype='int32')
        parameter_dict['curation']['searchAreaSize'] = np.array(parameter_dict['curation']['searchAreaSize'], dtype='int32')


        # NETWORK
        if parameter_dict['network']['backbone']['branchType'].lower().startswith('ResNet'.lower()):
            flavour = parameter_dict['network']['backbone']['branchType'].lower()[len('ResNet'):]

            if flavour == '18':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet18 as Backbone
            elif flavour == '34':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet34 as Backbone
            elif flavour == '50':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet50 as Backbone
            elif flavour == '101':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet101 as Backbone
            elif flavour == '152':
                from neuralNetwork.SiameseBranch_ResNet import SiameseBranch_ResNet152 as Backbone
            else:
                raise ValueError('ResNet type "{}" isn\'t supported!'.format(flavour))

        else:
            raise ValueError('Branch type "{}" isn\'t supported!'.format(parameter_dict['network']['backbone']['branchType']))

        parameter_dict['network']['backbone']['totalStride'] = np.array(Backbone.stride, dtype='int32')
        parameter_dict['network']['backbone']['filterSize'] = np.array(Backbone.filter_size, dtype='int32')

        parameter_dict['RPN']['numAnchors'] = len(parameter_dict['RPN']['anchorScales']) * \
                                              len(parameter_dict['RPN']['anchorRatios'])

        # TRACKER
        parameter_dict['Tracker']['modelFile'] = os.path.join(parameter_dict['Tracker']['modelPath'],
                                                              parameter_dict['Tracker']['modelName'])
        parameter_dict['Tracker']['trackingLogsFolder'] = os.path.join(parameter_dict['logs_folder'],
                                                                       parameter_dict['Tracker']['trackingLogsFolderName'])
        parameter_dict['Tracker']['scoreSize'] = get_tensor_size_after_convolution(
            tensor_size=parameter_dict['Tracker']['searchAreaSize'],
            filter_size=parameter_dict['Tracker']['exemplarSize'],
            stride=1,
            padding=0).astype("int32")

        parameter_dict['Tracker']['FCInference']['scaleFactors'] = np.array([parameter_dict['Tracker']['FCInference']['scaleStep'] ** i for i in range(int(np.ceil(-parameter_dict['Tracker']['FCInference']['numScales'] / 2.0)), int(np.floor(parameter_dict['Tracker']['FCInference']['numScales'] / 2.0 + 1)))])

        # Es el searchAreaSize y exemplarSize que tendria si se usaran imagenes como entrada en vez de features
        parameter_dict['Tracker']['virtualExemplarSize'] = get_tensor_size_before_convolution(
            feature_size=parameter_dict['Tracker']['exemplarSize'],
            filter_size=parameter_dict['network']['backbone']['filterSize'],
            stride=parameter_dict['network']['backbone']['totalStride'], padding=parameter_dict['network']['backbone']['padding'])
        parameter_dict['Tracker']['virtualSearchAreaSize'] = get_tensor_size_before_convolution(
            feature_size=parameter_dict['Tracker']['searchAreaSize'],
            filter_size=parameter_dict['network']['backbone']['filterSize'],
            stride=parameter_dict['network']['backbone']['totalStride'], padding=parameter_dict['network']['backbone']['padding'])

        # TRAINER
        parameter_dict['Trainer']['trainingLogsFolder'] = os.path.join(parameter_dict['logs_folder'],
                                                                       parameter_dict['Trainer']['trainingLogsFolderName'])
        parameter_dict['Trainer']['imdbEvaluationRatio'] = 1 - parameter_dict['Trainer']['imdbTrainingRatio']
        parameter_dict['Trainer']['scoreSize'] = get_tensor_size_after_convolution(
            tensor_size=parameter_dict['Trainer']['searchAreaSize'],
            filter_size=parameter_dict['Trainer']['exemplarSize'],
            stride=1,
            padding=0).astype("int32")

        # Es el searchAreaSize y exemplarSize que tendria si se usaran imagenes como entrada en vez de features
        parameter_dict['Trainer']['virtualExemplarSize'] = get_tensor_size_before_convolution(
            feature_size=parameter_dict['Trainer']['exemplarSize'],
            filter_size=parameter_dict['network']['backbone']['filterSize'],
            stride=parameter_dict['network']['backbone']['totalStride'], padding=parameter_dict['network']['backbone']['padding'])
        parameter_dict['Trainer']['virtualSearchAreaSize'] = get_tensor_size_before_convolution(
            feature_size=parameter_dict['Trainer']['searchAreaSize'],
            filter_size=parameter_dict['network']['backbone']['filterSize'],
            stride=parameter_dict['network']['backbone']['totalStride'], padding=parameter_dict['network']['backbone']['padding'])

        parameter_dict['curation']['trainingRawSize'] = parameter_dict['curation']['searchAreaSize'] + \
                                                        2 * parameter_dict['curation']['searchAreaAugmentationMargin']

        # =================== Movimiento de subdicts para mejor acceso ===================
        parameter_dict['network']['RPNLayer']['numAnchors'] = parameter_dict['RPN']['numAnchors']
        parameter_dict['network']['RPNLayer']['numAnchors'] = parameter_dict['RPN']['numAnchors']

        parameter_dict['Tracker']['network'] = copy.deepcopy(parameter_dict['network'])
        parameter_dict['Trainer']['network'] = copy.deepcopy(parameter_dict['network'])
        del parameter_dict['network']

        parameter_dict['Tracker']['Preprocessor']['padding'] = parameter_dict['Tracker']['network']['backbone']['padding']

        parameter_dict['Tracker']['RPNInference']['padding'] = parameter_dict['Tracker']['network']['backbone']['padding']
        parameter_dict['Trainer']['RPNTraining']['padding'] = parameter_dict['Trainer']['network']['backbone']['padding']

        parameter_dict['Tracker']['RPNInference']['RPN'] = copy.deepcopy(parameter_dict['RPN'])
        parameter_dict['Trainer']['RPNTraining']['RPN'] = copy.deepcopy(parameter_dict['RPN'])
        del parameter_dict['RPN']

        parameter_dict['Tracker']['randomSeed'] = parameter_dict['randomSeed']
        parameter_dict['Trainer']['randomSeed'] = parameter_dict['randomSeed']
        del parameter_dict['randomSeed']

        parameter_dict['Trainer']['FCTraining']['totalStride'] = copy.deepcopy(parameter_dict['Trainer']['network']['backbone']['totalStride'])
        parameter_dict['Trainer']['FCTraining']['filterSize'] = copy.deepcopy(parameter_dict['Trainer']['network']['backbone']['filterSize'])


        parameter_dict['Tracker']['network']['backbone']['exemplarSize'] = copy.deepcopy(parameter_dict['Tracker']['exemplarSize'])
        parameter_dict['Tracker']['network']['backbone']['searchAreaSize'] = copy.deepcopy(parameter_dict['Tracker']['searchAreaSize'])
        parameter_dict['Tracker']['network']['backbone']['contextAmount'] = parameter_dict['Tracker']['contextAmount']
        parameter_dict['Tracker']['network']['backbone']['cropExemplarFeatures'] = parameter_dict['Tracker']['cropExemplarFeatures']
        parameter_dict['Trainer']['network']['backbone']['exemplarSize'] = copy.deepcopy(parameter_dict['Trainer']['exemplarSize'])
        parameter_dict['Trainer']['network']['backbone']['searchAreaSize'] = copy.deepcopy(parameter_dict['Trainer']['searchAreaSize'])
        parameter_dict['Trainer']['network']['backbone']['contextAmount'] = parameter_dict['Trainer']['contextAmount']
        parameter_dict['Trainer']['network']['backbone']['cropExemplarFeatures'] = parameter_dict['Trainer']['cropExemplarFeatures']

        parameter_dict['Tracker']['network']['backbone']['loadAsOpenCV'] = parameter_dict['Tracker']['loadAsOpenCV']
        parameter_dict['Trainer']['network']['backbone']['loadAsOpenCV'] = parameter_dict['Trainer']['loadAsOpenCV']

        return parameter_dict

    def __init__(self, parameter_file=None, _parameter_dict=None):
        super(Parameters, self).__init__()

        if _parameter_dict is None:
            logger.debug('Loading default parameters')
            parameter_dict = Parameters.__default_dict()

        else:
            parameter_dict = _parameter_dict

        if parameter_file is not None:
            logger.debug('Loading file {} parameters'.format(parameter_file))
            custom_dict = Parameters.__read_json(os.path.abspath(parameter_file))
            extra_keys = Parameters.__json_update(json_old=parameter_dict, json_new=custom_dict)

            if len(extra_keys) > 0:
                logger.warning("Extra keys found in the JSON file: {}".format(extra_keys))

        if _parameter_dict is None:

            parameter_dict = Parameters.__get_precalculated(parameter_dict)

        for key, value in parameter_dict.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Parameters(_parameter_dict=x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Parameters(_parameter_dict=value) if isinstance(value, dict) else value)

    @staticmethod
    def __read_json(json_file):
        logger.debug('Reading json file')

        with open(json_file, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def __json_update(json_old, json_new, context=''):
        extra_keys = []

        # Compare all keys
        for key in json_new.keys():
            # if key exist in json2:
            if key in json_old.keys():
                # If subjson
                if type(json_new[key]) == dict:
                    logger.debug('Comparing subtree "{}"'.format('{}:{}'.format(context, key)))
                    extra_keys.extend(Parameters.__json_update(json_old=json_old[key],
                                                               json_new=json_new[key],
                                                               context='{}:{}'.format(context, key)))

                else:
                    logger.debug('Updating key "{}" from "{}" to "{}"'.format('{}:{}'.format(context, key), json_old[key], json_new[key]))
                    json_old[key] = json_new[key]

            else:
                logger.debug('Appending key "{}" = "{}"'.format('{}:{}'.format(context, key), json_new[key]))
                extra_keys.append('{}:{}'.format(context, key))
                json_old[key] = json_new[key]

        return extra_keys
