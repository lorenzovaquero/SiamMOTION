"""region_utils.py: Utilities for transforming region formats"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from tracking.AxisAlignedBB import AxisAlignedBB

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


def region_to_aabb(region):
    """Transforms a region (aka, each line inside a groundtruth file, in VOT's format)
    into an an Axis-Aligned Bounding-Box.
    Admits VOT's Rectangle and Polygon region format.
    The BB is defined by its center and its width and height"""

    element_number = len(region)
    assert (element_number == 4 or element_number == 8), (
        'Groundtruth\'s format must be a Rectange (4 elements) or a Polygon (8 elements)')

    if element_number is 4:
        return __rectangle_to_aabb(region)
    else:
        return __polygon_to_aabb(region)


"""471,230,28,88
    ^   ^   ^  ^
    |   |   |  L-- Altura (height)
    |   |   L----- Anchura (width)
    |   L--------- Coordenada superior (top)
    L------------- Coordenada izquierda (left)"""


def __rectangle_to_aabb(region):
    x = region[0]
    y = region[1]
    w = region[2]
    h = region[3]
    cx = x + w / 2
    cy = y + h / 2

    return AxisAlignedBB(cx, cy, w, h)


"""291.827,203.603,396.382,264.289,442.173,185.397,337.618,124.711
   ^--X--^ ^--Y--^ ^--X--^ ^--Y--^ ^--X--^ ^--Y--^ ^--X--^ ^--Y--^ 
   ^---Punto 1---^ ^---Punto 2---^ ^---Punto 3---^ ^---Punto 4---^"""


def __polygon_to_aabb(region, preserve_area=True):
    # OJO! En la implementacion original de SiamFC, la conversion de polygon a aabb se hace manteniendo constante
    # el area de la bbox resultante (preserve_area=True). No obstante, en VOT+TRAX, se hace simplemente con el
    # max/min de las esquinas de la bbox.
    if preserve_area:
        cx = np.mean(region[::2])
        cy = np.mean(region[1::2])
        x1 = np.min(region[::2])
        x2 = np.max(region[::2])
        y1 = np.min(region[1::2])
        y2 = np.max(region[1::2])
        A1 = np.linalg.norm(np.array(region[0:2]) - np.array(region[2:4])) * np.linalg.norm(
            np.array(region[2:4]) - np.array(region[4:6]))
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1

    else:
        x_values = region[0::2]
        y_values = region[1::2]

        xmin = np.amin(x_values)
        xmax = np.amax(x_values)
        ymin = np.amin(y_values)
        ymax = np.amax(y_values)

        w = xmax - xmin
        h = ymax - ymin
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

    return AxisAlignedBB(cx, cy, w, h)
