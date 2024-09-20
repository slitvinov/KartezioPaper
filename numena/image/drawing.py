"""

"""
import cv2
import numpy as np

from numena.enums import (
    IMAGE_UINT8_COLOR_1C,
    IMAGE_UINT8_COLOR_3C,
    IMAGE_UINT8_POSITIVE,
)
from numena.image.basics import image_new
from numena.image.contour import (
    contour_area,
    contours_convex_hull,
    contours_draw,
    contours_find,
    contours_merge,
)
from numena.io.image import imwrite


def draw_rectangle(image, x: int, y: int, width: int, height: int):
    """

    Parameters
    ----------
    image :
    x :
    y :
    width :
    height :
    """
    return cv2.rectangle(
        image, (x, y), (x + width, y + height), IMAGE_UINT8_COLOR_1C, -1
    )


def draw_overlay(image, mask, color=None, alpha=1.0, border_color="same", thickness=1):
    if color is None:
        color = IMAGE_UINT8_COLOR_3C
    out = image.copy()
    img_layer = image.copy()
    img_layer[np.where(mask)] = color
    overlayed = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    contours = contours_find(mask, exclude_holes=False)
    if border_color == "same":
        overlayed = contours_draw(overlayed, contours, thickness=thickness, color=color)
    elif border_color is not None:
        overlayed = contours_draw(
            overlayed, contours, thickness=thickness, color=border_color
        )
    return overlayed


def fill_ellipses(mask, ellipses, color=IMAGE_UINT8_POSITIVE):
    for ellipse in ellipses:
        cv2.ellipse(mask, ellipse, color, thickness=-1)
    return mask


def fill_ellipses_as_labels(mask, ellipses):
    for i, ellipse in enumerate(ellipses):
        cv2.ellipse(mask, ellipse, i + 1, thickness=-1)
    return mask


def fill_polygons(mask, polygons, color=IMAGE_UINT8_POSITIVE):
    return cv2.fillPoly(mask, pts=polygons, color=color)


def fill_polygons_as_labels(mask, polygons):
    for i, polygon in enumerate(polygons):
        cv2.fillPoly(mask, pts=np.int32([polygon]), color=i + 1)
    return mask


if __name__ == "__main__":
    image_test = image_new((512, 512))
    image_test = draw_rectangle(image_test, 256, 256, 64, 64)
    image_test = draw_rectangle(image_test, 16, 16, 64, 64)
    image_test = draw_rectangle(image_test, 92, 16, 16, 64)
    imwrite("rectangle_test.png", image_test)

    contours = contours_find(image_test)
    for cnt in contours:
        area = contour_area(cnt)
        print(area)

    merged_contour = contours_merge(contours)
    convex_hull = contours_convex_hull(merged_contour)
    area = contour_area(convex_hull)
    print(area)
    image_test = contours_draw(image_test, [convex_hull], color=[128], thickness=3)
    imwrite("rectangle_test_merged.png", image_test)
