"""select_region.py: Utilities for defining regions over images and videos"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, CloseEvent

from .RescalableBoundingBoxCreator import RescalableBoundingBoxCreator
from .VariableZoomCreator import VariableZoomCreator

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


IS_CANVAS_OPEN = False


def select_region(image):
    """Opens a window to let the user select a region, and returns the region"""
    figure = plt.figure("Click and drag a Bounding Box over the image and press Enter")
    figure.canvas.mpl_connect('key_press_event', __kill_playing)

    axis = figure.add_subplot(111)

    axis.imshow(image)

    z = VariableZoomCreator(axis, scale_change=1.5)

    bb = RescalableBoundingBoxCreator(axis)

    plt.show()

    left = min(bb.x0, bb.x1)
    top = min(bb.y0, bb.y1)
    width = abs(bb.x0 - bb.x1)
    height = abs(bb.y0 - bb.y1)

    return left, top, width, height


def select_region_live(video):
    """Opens a window to let the user select a region during a video, and returns the region"""

    global IS_CANVAS_OPEN
    IS_CANVAS_OPEN = True

    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    x_lim = None
    y_lim = None

    while IS_CANVAS_OPEN:
        video.next_frame()

        try:
            figure = plt.figure("Click and drag a Bounding Box over the video and press Enter")

        except:  # If the video was closed pressing the "x"
            break

        figure.canvas.mpl_connect('key_press_event', __kill_playing)
        figure.canvas.mpl_connect('close_event', __kill_playing)

        axis = figure.add_subplot(111)

        bb = RescalableBoundingBoxCreator(axis, x0, y0, x1, y1)

        axis.imshow(cv2.cvtColor(video.current_frame, cv2.COLOR_BGR2RGB))

        z = VariableZoomCreator(axis, scale_change=1.5, x_lim=x_lim, y_lim=y_lim)

        plt.ion()
        plt.show()

        plt.pause(0.000001)
        plt.clf()

        x0 = bb.x0
        y0 = bb.y0
        x1 = bb.x1
        y1 = bb.y1

        x_lim = z.x_lim
        y_lim = z.y_lim

    plt.close("all")

    left = min(bb.x0, bb.x1)
    top = min(bb.y0, bb.y1)
    width = abs(bb.x0 - bb.x1)
    height = abs(bb.y0 - bb.y1)

    return left, top, width, height


def __kill_playing(event):
    """When Escape is pressed or the figure is closed"""
    global IS_CANVAS_OPEN

    if type(event) == KeyEvent:
        if event.key == 'enter' or event.key == 'escape':
            IS_CANVAS_OPEN = False
            plt.close(event.canvas.figure)

    elif type(event) == CloseEvent:
        IS_CANVAS_OPEN = False
