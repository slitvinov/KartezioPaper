"""Contours Module

"""

import cv2
import numpy as np

from numena.enums import IMAGE_UINT8_COLOR_1C, IMAGE_UINT8_COLOR_3C


def contours_convex_hull(contour):
    return cv2.convexHull(contour)


def contours_merge(contours):
    return np.vstack(contours)


def contour_area(contour):
    return cv2.contourArea(contour)


def contours_find(image, exclude_holes=True):
    mode = cv2.RETR_LIST
    method = cv2.CHAIN_APPROX_SIMPLE
    if exclude_holes:
        mode = cv2.RETR_EXTERNAL
    return cv2.findContours(image.copy(), mode, method)[0]


def _draw_contours(
    image, contours, color=None, selected: int = None, fill=True, thickness=1
):
    assert (
        len(image.shape) == 3 or len(image.shape) == 2
    ), "given image wrong format, shape must be (h, w, c) or (h, w)"
    if color is None:
        if len(image.shape) == 3 and image.shape[-1] == 3:
            color = IMAGE_UINT8_COLOR_3C
        elif len(image.shape) == 2:
            color = IMAGE_UINT8_COLOR_1C
        else:
            raise ValueError("Image wrong format, must have 1 or 3 channels")
    if selected is None:
        selected = -1  # selects all the contours (-1)
    if fill:
        thickness = -1  # fills the contours (-1)
    return cv2.drawContours(image.copy(), contours, selected, color, thickness)


def contours_fill(image, contours, color=None, selected=None):
    return _draw_contours(image, contours, color, selected, fill=True)


def contours_draw(image, contours, color=None, selected=None, thickness=1):
    return _draw_contours(
        image, contours, color, selected, fill=False, thickness=thickness
    )
