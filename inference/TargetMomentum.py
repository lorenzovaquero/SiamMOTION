"""TargetMomentum.py: Data container for calculating the momentum of a target"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import collections

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from tracking.AxisAlignedBB import AxisAlignedBB

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2018"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzovaquero@hotmail.com"
__status__ = "Prototype"


class TargetMomentum(object):
    """Applies momentum to a target, based on its previous positions"""

    def __init__(self, target, searchAreaCropSize, maxSize=3, minUpdates=2):
        self.maxSize = maxSize
        self.minUpdates = minUpdates
        self.factorial = self.maxSize * (self.maxSize + 1) / 2

        self.lastTarget = target
        self.targetHistoryX = collections.deque([], maxSize)
        self.targetHistoryY = collections.deque([], maxSize)

        self.lastSearchAreaCropSize = searchAreaCropSize
        self.searchAreaCropSizeHistory = collections.deque([], maxSize)

    def updateTarget(self, target, searchAreaCropSize):

        self.targetHistoryX.append(target.centerX - self.lastTarget.centerX)
        self.targetHistoryY.append(target.centerY - self.lastTarget.centerY)
        self.lastTarget = target

        sameSignX = 0
        previousTargetX = self.targetHistoryX[len(self.targetHistoryX) - 1]
        for i in reversed(self.targetHistoryX):
            if np.sign(previousTargetX) == np.sign(i):
                sameSignX += 1
            else:
                break

        totalX = 0
        if sameSignX >= self.minUpdates:
            for i, j in reversed(list(enumerate(self.targetHistoryX))):
                totalX += (i + 1) * j
            totalX /= self.factorial
        else:
            totalX = 0

        sameSignY = 0
        previousTargetY = self.targetHistoryY[len(self.targetHistoryY) - 1]
        for i in reversed(self.targetHistoryY):
            if np.sign(previousTargetY) == np.sign(i):
                sameSignY += 1
            else:
                break

        totalY = 0
        if sameSignY >= self.minUpdates:
            for i, j in reversed(list(enumerate(self.targetHistoryY))):
                totalY += (i + 1) * j
            totalY /= self.factorial
        else:
            totalY = 0

        self.searchAreaCropSizeHistory.append(searchAreaCropSize - self.lastSearchAreaCropSize)
        self.lastSearchAreaCropSize = searchAreaCropSize

        sameSignSearchAreaCropSize = 0
        previousSearchAreaCropSize = self.searchAreaCropSizeHistory[len(self.searchAreaCropSizeHistory) - 1]
        for i in reversed(self.searchAreaCropSizeHistory):
            if np.sign(previousSearchAreaCropSize) == np.sign(i):
                sameSignSearchAreaCropSize += 1
            else:
                break

        totalSearchAreaCropSize = 0
        if sameSignSearchAreaCropSize >= self.minUpdates:
            for i, j in reversed(list(enumerate(self.searchAreaCropSizeHistory))):
                totalSearchAreaCropSize += (i + 1) * j
            totalSearchAreaCropSize /= self.factorial
        else:
            totalSearchAreaCropSize = 0

        return AxisAlignedBB(target.centerX + totalX, target.centerY + totalY, target.width,
                             target.height), searchAreaCropSize + totalSearchAreaCropSize
