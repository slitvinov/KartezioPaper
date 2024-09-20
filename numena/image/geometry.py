import cv2
import numpy as np

from numena.enums import IMAGE_UINT8_POSITIVE
from numena.image.basics import image_new
from numena.image.contour import contours_find


def intersection_with_line(mask, line):
    line_mask = image_new(mask.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], IMAGE_UINT8_POSITIVE, 2)
    contours = contours_find(mask)
    mask_cnt = image_new(mask.shape)
    cv2.drawContours(mask_cnt, contours, -1, IMAGE_UINT8_POSITIVE, 2)
    intersection = cv2.bitwise_and(line_mask, mask_cnt)
    centroid = np.mean(np.argwhere(intersection), axis=0)
    return centroid
