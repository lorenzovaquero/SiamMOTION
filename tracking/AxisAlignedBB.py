"""AxisAlignedBB.py: Data container for an axis-aligned bounding box"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class AxisAlignedBB(object):
    """Data container for the size and location of a target"""

    def __init__(self, center_x, center_y, width, height):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

        self.left_top_point = None
        self.right_top_point = None
        self.left_bottom_point = None
        self.right_bottom_point = None

        self.top = None
        self.bottom = None
        self.right = None
        self.left = None

        self.center = None
        self.size = None

        self.tensor = np.array([self.center_y, self.center_x, self.height, self.width])  # center_vertical, center_horizontal, height, width

        self.__calculate_points()

    def __calculate_points(self):
        self.top = self.center_y - (self.height - 1) / 2
        self.bottom = self.center_y + (self.height - 1) / 2
        self.left = self.center_x - (self.width - 1) / 2
        self.right = self.center_x + (self.width - 1) / 2

        self.left_top_point = np.array((self.left, self.top))
        self.right_top_point = np.array((self.right, self.top))
        self.left_bottom_point = np.array((self.left, self.bottom))
        self.right_bottom_point = np.array((self.right, self.bottom))

        self.center = np.array([self.center_x, self.center_y])
        self.size = np.array([self.width, self.height])

    @classmethod
    def from_polygon(cls, region, preserve_area=True):
        # OJO! En la implementacion original de SiamFC, la conversion de polygon a aabb se hace manteniendo constante
        # el area de la bbox resultante (preserve_area=True). No obstante, en VOT+TRAX, se hace simplemente con el
        # max/min de las esquinas de la bbox.

        try:
            region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                               region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
        except:
            region = np.array(region)

        if preserve_area:
            cx = np.mean(region[0::2])
            cy = np.mean(region[1::2])
            x1 = min(region[0::2])
            x2 = max(region[0::2])
            y1 = min(region[1::2])
            y2 = max(region[1::2])
            A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
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

        return cls(center_x=cx, center_y=cy, width=w, height=h)