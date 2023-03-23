"""VariableZoomCreator.py: Allows the use of zoom in plots using the scroll wheel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class VariableZoomCreator(object):
    """Allows the use of zoom with the scroll wheel"""

    def __init__(self, axis, scale_change=2.0, x_lim=None, y_lim=None):
        self.pressed = False
        self.ax = axis
        self.scaleChange = scale_change

        if x_lim is None or y_lim is None:
            self.x_lim = self.ax.get_xlim()
            self.y_lim = self.ax.get_ylim()

        else:
            self.x_lim = x_lim
            self.y_lim = y_lim

            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)

        self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        """When the user scrolls the mouse wheel"""

        # get the current x and y limits
        self.x_lim = self.ax.get_xlim()
        self.y_lim = self.ax.get_ylim()

        x_pos = event.xdata  # get event x location
        y_pos = event.ydata  # get event y location

        if x_pos is None or y_pos is None:
            x_pos = (self.x_lim[1] + self.x_lim[0]) / 2
            y_pos = (self.y_lim[0] + self.y_lim[1]) / 2

        if event.button == 'up':
            scale_factor = self.scaleChange

        elif event.button == 'down':
            scale_factor = 1 / self.scaleChange

        else:
            # deal with something that should never happen
            scale_factor = 1

        # set new limits
        self.ax.set_xlim([x_pos - (x_pos - self.x_lim[0]) / scale_factor, x_pos + (self.x_lim[1] - x_pos) / scale_factor])
        self.ax.set_ylim([y_pos - (y_pos - self.y_lim[0]) / scale_factor, y_pos + (self.y_lim[1] - y_pos) / scale_factor])

        self.ax.figure.canvas.draw()  # plt.draw()  # force re-draw
