"""Threshold Module

"""

import cv2

from numena.enums import IMAGE_UINT8_POSITIVE


def threshold_binary(image, threshold, value=IMAGE_UINT8_POSITIVE):
    return cv2.threshold(image, threshold, value, cv2.THRESH_BINARY)[1]


def threshold_tozero(image, threshold):
    return cv2.threshold(image, threshold, IMAGE_UINT8_POSITIVE, cv2.THRESH_TOZERO)[1]
