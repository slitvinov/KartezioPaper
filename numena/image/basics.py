"""Image Module

"""

import cv2
import numpy as np


def image_new(shape, dtype=np.uint8):
    return np.zeros(shape=shape, dtype=dtype)


def image_like(image):
    return np.zeros_like(image)


def image_split(image):
    return list(cv2.split(image))


def image_ew_mean(image_1, image_2):
    return cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)


def image_ew_max(image_1, image_2):
    return cv2.max(image_1, image_2)


def image_ew_min(image_1, image_2):
    return cv2.min(image_1, image_2)


def image_normalize(image):
    return cv2.normalize(
        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
