"""parameters.py: Parameter description and management"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import logging

logger = logging.getLogger(__name__)


__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2018"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzovaquero@hotmail.com"
__status__ = "Prototype"


def get_parameters(parameter_file=None):
    logger.debug('Reading default parameters')

    params = {}


    # ------------------------------------------------------------------------------------------------------------------
    #####################################################################
    # GENERAL PARAMETERS
    #####################################################################

    params['gpuId'] = 0  # integer [-1, +inf) | id of the GPU used for computations ("-1" means auto-select)
    params['logs_folder'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))  # path | folder for storing tracking and training logs
    params['branchType'] = 'AlexNet'  # text ['AlexNet' OR 'AlexNet'] | type of neural network that will make up the branches
    params['totalStride'] = 8  # integer [1, +inf) | total stride of the network, obtained by multiplying the individual strides of each layer
    params['exemplarSize'] = 127  # integer [1, +inf) | side size (in pixels) of the square exemplar image that will enter the network
    params['searchAreaSize'] = 255  # integer [1, +inf) | side size (in pixels) of the square search area images that will enter the network
    params['contextAmount'] = 0.5  # float [0, +inf) | amount of context left around a bounding box for obtaining its exemplar image
    params['adjustFactor'] = 0.001  # float (0, +inf) | value of the adjust convolutional layer, in order to tune the loss calculation
    params['epsilon'] = 1e-6  # float (0, +inf) | value used in order to avoid zeroes in the variance calculation

    # ------------------------------------------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------------------------------------------
    #####################################################################
    # TRACKER PARAMETERS
    #####################################################################

    params['numScales'] = 3  # integer [1, +inf) | number of scales that will be considered during tracking (only odd numbers will make a change) | a "5" would mean that there are considered 2 bigger scales, 2 smaller scales and the same scale
    params['scaleStep'] = 1.0375  # float (1, +inf) | step of the change in size when searching over scales | it will be the base of an exponentiation, so it souldn't be more than "1.5"
    params['scalePenalty'] = 0.9745  # float [0, 1] | penalization factor applied to the score maps of a different scale
    params['scaleDamping'] = 0.59  # float [0, 1] | factor for updating the scale by linear interpolation, to reduce damping. Reduces the effect of the 'scaleStep' when updating the bounding box
    params['sizeMinFactor'] = 0.2  # float [0, 1] | target's lower size limit. It's a factor that multiplies the target's initial size, and prevents shrinking the object if it gets smaller than that
    params['sizeMaxFactor'] = 5.0  # float [1, +inf) | target's upper size limit. It's a factor that multiplies the target's initial size, and prevents enlarging the object if it gets bigger than that
    params['upsampleFactor'] = 16  # integer [1, +inf) | factor for upsampling the score map with bicubic interpolation, in order to increase the accuracy, making it less coarse
    params['windowing'] = 'cosine'  # text ['cosine' OR 'uniform'] | stablishes the distribution of the weights at the displacement penalization window | a "uniform" value will cancel any displacement penalization
    params['windowInfluence'] = 0.176  # float [0, 1] | influence of the displacement window (in convex sum)

    params['modelPath'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))  # path | folder where the trained model is stored
    params['modelName'] = 'model_tf.ckpt'  # text | name of the trained model file
    params['trackingLogsFolderName'] = 'tracking'  # text | name of the subfolder where the tracking logs will be stored
    params['trackingLogsStep'] = 100  # integer [1, +inf) | number of frames run between log writes at tracking

    # ------------------------------------------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------------------------------------------
    #####################################################################
    # TRAINING PARAMETERS
    #####################################################################

    params['randomSeed'] = 1  # integer [0, 4294967295] | seed used for generating the random numbers during training (it allows result replication)
    params['trainBatchSize'] = 8  # integer [1, +inf) | number of image pairs that will be processed at each step (at the same time) during training
    params['imdbTrainingRatio'] = 1  # float [0, 1] | percentage of imdb videos that wil be used for training (the rest will be used for evaluation)
    params['searchAreaAugmentationMargin'] = 8  # integer [0, 'searchAreaSize'/2) | margin (in pixels) left in the search area images used for training in order to allow the augmentation
    params['positiveRadius'] = 16  # integer [1, 'totalStride' * 'scoreSize' * sqrt(2) / 2] | radius of pixels, centered at the target which will be given a positive label during training
    params['neutralRadius'] = 80 # integer [0, 'totalStride' * 'scoreSize' * sqrt(2) / 2 - 'positiveRadius'] | radius of pixels, centered at the target which will be given a neutral label during training (the positive radius won't be altered)
    params['numPairs'] = 53200  # integer [1, +inf) | number of image pairs that will be processed at each epoch during training
    params['trainNumEpochs'] = 50  # integer [1, +inf) | number of epochs that will be done during training (each epoch will process 'numPairs' image pairs)
    params['frameRange'] = 100  # integer [1, +inf) | maximum separation (in frames) between an exemplar image and its search area during training
    params['initMethod'] = 'kaiming'  # text ['kaiming' OR 'xavier'] | method used for initializing the weights during training

    params['momentum'] = 0.9  # float (0, +inf) | amount of momentum applied during training when computing gradients
    params['stddev'] = 0.01  # float [0, +inf) | Standard deviation of the normal distribution when initializing net weights
    params['trainLearningRateStart'] = 1e-2  # float (0, +inf) | value of the learning rate at the start of the training
    params['trainLearningRateStop'] = 1e-5  # float (0, +inf) | value of the learning rate at the end of the training
    params['trainLearningRatePolicy'] = 'exponential'  # text ['exponential' OR 'cosine' OR 'linear'] | annealing policy used for decaying the learning rate during training
    params['convolutionWeightDecay'] = 5e-4  # float (0, 1] | weight decay applied at convolutions during training, where after each update, the weights are multiplied by this factor in order to prevent them from growing too large
    params['batchNormalizationWeightDecay'] = 0.95  # float (0, 1] | weight decay applied at batch normalizations during training, where after each update, the weights are multiplied by this factor in order to prevent them from growing too large

    params['maxStretch'] = 0.05  # float [0, 1] | maximum amount of stretch applied when altering images for training
    params['maxTranslate'] = 4  # integer [0, searchAreaAugmentationMargin] | maximum amount of translation applied when altering images for training
    params['colorVarianceFactor'] = 0.05  # float [0, +inf) | influence of the dataset color standard deviation when altering images for training
    params['mirroringProbability'] = 0  # float [0, 1] | probability of mirroring a frame when altering images for training

    params['trainingLogsFolderName'] = 'training'  # text | name of the subfolder where the training logs will be stored
    params['trainingLogsStep'] = 100  # integer [1, +inf) | number of batches run between log writes at training

    # ------------------------------------------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------------------------------------------
    #####################################################################
    # VALIDATION PARAMETERS
    #####################################################################

    params['imageLogsStep'] = 2500  # integer [1, +inf) | number of batches run between log writes at validation

    # ------------------------------------------------------------------------------------------------------------------




    # We could override some of the parameters with the contents of a json file
    if parameter_file is not None:
        json_data = __read_json(parameter_file)
        params.update(json_data)



    # ------------------------------------------------------------------------------------------------------------------
    #####################################################################
    # CALCULATED PARAMETERS
    #####################################################################

    # General

    params['scoreSize'] = (params['searchAreaSize'] - params['exemplarSize']) / params['totalStride'] + 1  # integer | Size of the score map that results from cross-correlate the searh area and exemplar features

    # Tracking
    params['trackingLogsFolder'] = os.path.join(params['logs_folder'], params['trackingLogsFolderName'])  # text | path of the folder where the tracking logs will be stored
    params['modelFile'] = os.path.join(params['modelPath'], params['modelName'])  # text | path of the trained model file used for tracking

    # Training
    params['trainingLogsFolder'] = os.path.join(params['logs_folder'], params['trainingLogsFolderName'])  # text | path of the folder where the training logs will be stored
    params['imdbEvaluationRatio'] = 1 - params['imdbTrainingRatio']  # float | percentage of imdb videos that wil be used for evaluation (the rest will be used for training)
    params['trainingSearchAreaSize'] = params['searchAreaSize'] - 2 * params['searchAreaAugmentationMargin']  # integer | size of the search area images used for training (if the raw image is not directly used), after the augmentation is applied
    params['trainingRawSize'] = params['searchAreaSize'] + 2 * params['searchAreaAugmentationMargin']  # integer | size of the raw images saved in order to have 'searchAreaSize' size images during training after data augmentation

    # ------------------------------------------------------------------------------------------------------------------


    return params


def __read_json(json_file):
    logger.debug('Reading parameter file')

    with open(json_file, 'r') as f:
        data = json.load(f)

    return data