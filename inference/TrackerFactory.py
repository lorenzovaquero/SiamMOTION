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


def get_tracker(flavour='DSARPN'):
    if flavour.upper() == 'RPN'.upper():
        from inference.TrackerSiamMT_RPN import TrackerSiamMT_RPN
        tracker = TrackerSiamMT_RPN

    elif flavour.upper() == 'DSARPN'.upper():
        from inference.TrackerSiamMT_DSARPN import TrackerSiamMT_DSARPN
        tracker = TrackerSiamMT_DSARPN

    else:
        raise ValueError('Tracker type "{}" isn\'t supported!'.format(flavour))

    return tracker
