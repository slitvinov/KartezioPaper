"""Morphology Module

"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from numena.image.contour import contours_fill, contours_find


class WatershedTransform:
    pass


class WatershedSkimage(WatershedTransform):
    def __init__(self, use_dt=False, markers_distance=21, markers_area=None):
        self.use_dt = use_dt
        self.markers_distance = markers_distance
        self.markers_area = markers_area

    def _extract_markers(self, signal):
        peak_idx = peak_local_max(
            signal,
            min_distance=self.markers_distance,
            exclude_border=0,
            # footprint=np.ones((21, 21)),
            num_peaks=255,
        )
        peak_mask = np.zeros_like(signal, dtype=np.uint8)
        peak_mask[tuple(peak_idx.T)] = 1
        return peak_mask

    def apply(self, signal, markers=None, mask=None):
        if self.use_dt:
            signal = ndimage.distance_transform_edt(signal)
        if markers is None:
            markers = self._extract_markers(signal)

        # markers[mask == 0] = 0
        if self.markers_area:
            n, marker_labels, stats, _ = cv2.connectedComponentsWithStats(
                markers, connectivity=8
            )
            for i in range(1, n):
                if (
                    self.markers_area[0]
                    < stats[i, cv2.CC_STAT_AREA]
                    < self.markers_area[1]
                ):
                    pass
                else:
                    marker_labels[marker_labels == i] = 0
        else:
            marker_labels = cv2.connectedComponents(
                markers, connectivity=8, ltype=cv2.CV_16U
            )[1]

        # limit the number of markers to 255
        marker_labels[marker_labels > 255] = 0
        signal_inv = 255 - signal
        labels = watershed(
            signal_inv, markers=marker_labels, mask=mask, watershed_line=True
        )
        return signal, marker_labels, labels


def get_kernel(struct_shape="circle", half_kernel_size=1):
    kernel_size = half_kernel_size * 2 + 1
    if struct_shape == "circle":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif struct_shape == "square":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif struct_shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))


def morph_erode(image, struct_shape="circle", half_kernel_size=1):
    kernel = get_kernel(struct_shape=struct_shape, half_kernel_size=half_kernel_size)
    return cv2.erode(image, kernel)


def morph_dilate(image, struct_shape="circle", half_kernel_size=1):
    kernel = get_kernel(struct_shape=struct_shape, half_kernel_size=half_kernel_size)
    return cv2.dilate(image, kernel)


def morph_fill(image):
    cnts = contours_find(image, exclude_holes=True)
    return contours_fill(image, cnts)
