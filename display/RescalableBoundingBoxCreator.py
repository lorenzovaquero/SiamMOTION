"""RescalableBoundingBoxCreator.py: Bounding box that can be rescaled by point-and-click"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.patches import Rectangle

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class RescalableBoundingBoxCreator(object):
    """Bounding box that can be rescaled by point-and-click"""

    def __init__(self, axis, x0=0, y0=0, x1=0, y1=0):
        self.pressed = False
        self.ax = axis
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.rect = Rectangle((self.x0, self.y0), self.x1 - self.x0, self.y1 - self.y0, linewidth=2, edgecolor='y', fill=False)

        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """When user presses the mouse button"""
        self.pressed = True
        if event.xdata is not None:
            self.x0 = event.xdata

        if event.ydata is not None:
            self.y0 = event.ydata

    def on_release(self, event):
        """When user releases the mouse button"""
        self.pressed = False
        if event.xdata is not None:
            self.x1 = event.xdata

        if event.ydata is not None:
            self.y1 = event.ydata

        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

    def on_motion(self, event):
        """When user moves the mouse"""
        if self.pressed:

            if event.xdata is not None:
                self.x1 = event.xdata

            if event.ydata is not None:
                self.y1 = event.ydata

            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.ax.figure.canvas.draw()
