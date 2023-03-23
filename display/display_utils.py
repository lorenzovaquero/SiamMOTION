"""display_utils.py: Utils for joining images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


def horizontal_stack(left_image, right_image, separation=5, bgr_color=tuple([255, 255, 255])):
    lh, lw, _ = left_image.shape
    rh, rw, _ = right_image.shape

    if lh > rh:
        difference = lh - rh
        if difference % 2 == 0:
            top_padding = difference // 2
            bottom_padding = top_padding
        else:
            top_padding = difference // 2
            bottom_padding = top_padding + 1

        right_image = cv2.copyMakeBorder(right_image, top_padding, bottom_padding, separation, 0, cv2.BORDER_CONSTANT, value=bgr_color)

    elif lh < rh:
        difference = rh - lh
        if difference % 2 == 0:
            top_padding = difference // 2
            bottom_padding = top_padding
        else:
            top_padding = difference // 2
            bottom_padding = top_padding + 1

        left_image = cv2.copyMakeBorder(left_image, top_padding, bottom_padding, 0, separation, cv2.BORDER_CONSTANT, value=bgr_color)

    else:
        left_image = cv2.copyMakeBorder(left_image, 0, 0, 0, separation, cv2.BORDER_CONSTANT, value=bgr_color)

    image = np.hstack((left_image, right_image))

    return image


def vertical_stack(top_image, bottom_image, separation=5, bgr_color=tuple([255, 255, 255])):
    th, tw, _ = top_image.shape
    bh, bw, _ = bottom_image.shape

    if tw > bw:
        difference = tw - bw
        if difference % 2 == 0:
            left_padding = difference // 2
            right_padding = left_padding
        else:
            left_padding = difference // 2
            right_padding = left_padding + 1

        bottom_image = cv2.copyMakeBorder(bottom_image, separation, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=bgr_color)

    elif tw < bw:
        difference = bw - tw
        if difference % 2 == 0:
            left_padding = difference // 2
            right_padding = left_padding
        else:
            left_padding = difference // 2
            right_padding = left_padding + 1

        top_image = cv2.copyMakeBorder(top_image, 0, separation, left_padding, right_padding, cv2.BORDER_CONSTANT, value=bgr_color)

    else:
        top_image = cv2.copyMakeBorder(top_image, 0, separation, 0, 0, cv2.BORDER_CONSTANT, value=bgr_color)

    image = np.vstack((top_image, bottom_image))

    return image


def create_value_image(label, value, txt_color=tuple([0, 0, 0]), bgr_color=tuple([255, 255, 255]), height=35,
                       label_width=230, value_width=190, text_gap=10, label_scale=1.05, label_thickness=1,
                       value_scale=1.05, value_thickness=1):
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, label_scale, label_thickness)

    label_blue_channel = np.full((height, label_width), bgr_color[0]).astype(np.uint8)
    label_green_channel = np.full((height, label_width), bgr_color[1]).astype(np.uint8)
    label_red_channel = np.full((height, label_width), bgr_color[2]).astype(np.uint8)
    label_blank = np.stack((label_blue_channel, label_green_channel, label_red_channel), axis=2)

    cv2.putText(label_blank, label, tuple([label_width - label_size[0][0], int((height / 2) + (label_size[0][1] / 2))]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, label_scale, txt_color, label_thickness, cv2.LINE_AA)

    value_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_COMPLEX_SMALL, value_scale, value_thickness)
    value_blue_channel = np.full((height, value_width), bgr_color[0]).astype(np.uint8)
    value_green_channel = np.full((height, value_width), bgr_color[1]).astype(np.uint8)
    value_red_channel = np.full((height, value_width), bgr_color[2]).astype(np.uint8)
    value_blank = np.stack((value_blue_channel, value_green_channel, value_red_channel), axis=2)

    cv2.putText(value_blank, value, tuple([0, int((height / 2) + (value_size[0][1] / 2))]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                value_scale, txt_color, value_thickness, cv2.LINE_AA)

    image = horizontal_stack(label_blank, value_blank, separation=text_gap, bgr_color=bgr_color)
    return image


def transform_matrix_range(input_matrix, new_max, new_min, copy=False):
    if copy:
        matrix = input_matrix.copy()
    else:
        matrix = input_matrix

    matrix += -(np.min(matrix))

    # To avoid nan
    if np.max(matrix) == 0 and np.min(matrix) == 0:
        matrix = np.full_like(matrix, new_min)
    elif np.max(matrix) == 0:
        matrix /= np.nextafter(0, 1) / (new_max - new_min)
    else:
        matrix /= np.max(matrix) / (new_max - new_min)

    matrix += new_min
    return matrix

def features_to_image(features):
    image = transform_matrix_range(features[:, :, 0:3], 255, 0, copy=True).astype(np.uint8)  # TODO: Ver si sumar o algo

    return image



def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y = int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def drawpoly(img, pts, color, thickness=1, style='dotted', gap=20):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style, gap)

def drawrect(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style, gap)
