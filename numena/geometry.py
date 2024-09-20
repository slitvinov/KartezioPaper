from __future__ import annotations

from dataclasses import asdict, astuple, dataclass
from math import atan2, cos, degrees, pi, radians, sin
from typing import ClassVar, Tuple

import cv2
import numpy as np

from numena.image.contour import contours_fill, contours_find
from numena.image.geometry import intersection_with_line
from numena.image.morphology import morph_dilate, morph_erode


@dataclass
class Vector2:
    x: float
    y: float
    ZERO: ClassVar[Vector2]
    UP: ClassVar[Vector2]
    DOWN: ClassVar[Vector2]
    LEFT: ClassVar[Vector2]
    RIGHT: ClassVar[Vector2]

    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(self.x - other.x, self.y - other.y)

    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)

    def dot(self, other: Vector2) -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return self.dot(self) ** 0.5

    def normalized(self) -> Vector2:
        norm = self.norm()
        return Vector2(self.x / norm, self.y / norm)

    def distance(self, other: Vector2) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def perp(self) -> Vector2:
        return Vector2(1, -self.x / self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    def as_int_tuple(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    def angle_with_x_axis(self, other: Vector2) -> float:
        diff = other - self
        rad = atan2(diff.y, diff.x)
        if rad < 0:
            rad += 2 * pi
        return degrees(rad)

    def __str__(self) -> str:
        return str(astuple(self))

    def __repr__(self) -> str:
        return f"Vector2 {asdict(self)}"


Vector2.ZERO = Vector2(0.0, 0.0)
Vector2.UP = Vector2(0.0, 1.0)
Vector2.DOWN = Vector2(0.0, -1.0)
Vector2.LEFT = Vector2(-1.0, 0.0)
Vector2.RIGHT = Vector2(1.0, 0.0)


@dataclass
class Line2:
    pt1: Vector2
    pt2: Vector2
    ZERO: ClassVar[Line2]
    UP: ClassVar[Line2]
    DOWN: ClassVar[Line2]
    LEFT: ClassVar[Line2]
    RIGHT: ClassVar[Line2]

    @staticmethod
    def from_point(
        point: Vector2, angle: float, size: float, centered: bool = False
    ) -> Line2:
        angle_rad = radians(angle)
        angle_rad_inv = radians(angle - 180)
        if centered:
            size = size / 2
            point_1 = point + Vector2(cos(angle_rad_inv), sin(angle_rad_inv)) * size
        else:
            point_1 = point
        point_2 = point + Vector2(cos(angle_rad), sin(angle_rad)) * size
        return Line2(point_1, point_2)

    def as_int_tuple(self):
        return self.pt1.as_int_tuple(), self.pt2.as_int_tuple()


Line2.ZERO = Line2(Vector2.ZERO, Vector2.ZERO)
Line2.UP = Line2(Vector2.ZERO, Vector2.UP)
Line2.DOWN = Line2(Vector2.ZERO, Vector2.DOWN)
Line2.LEFT = Line2(Vector2.ZERO, Vector2.LEFT)
Line2.RIGHT = Line2(Vector2.ZERO, Vector2.RIGHT)


class MicroEntity:
    def __init__(self, name, custom_data=None):
        self.name = name
        self.custom_data = custom_data
        if self.custom_data is None:
            self.custom_data = {}

    def set_data(self, data_name, data):
        self.custom_data[data_name] = data

    def get_data(self, data_name):
        return self.custom_data[data_name]


class MicroEntity2D(Vector2, MicroEntity):
    def __init__(self, name, mask, x, y, custom_data=None):
        MicroEntity.__init__(self, name, custom_data=custom_data)
        Vector2.__init__(self, x, y)
        self.mask = mask

    def get_mean(self, channel):
        return np.mean(channel, where=self.mask > 0)

    def get_sum(self, channel, threshold=0):
        return np.sum(channel, where=self.mask > threshold)

    def get_max(self, channel):
        return np.max(channel, where=self.mask > 0, initial=0)

    @property
    def area(self):
        return cv2.countNonZero(self.mask)

    @property
    def boundary(self):
        if self.area == 0:
            return None
        return contours_find(self.mask)

    @property
    def perimeter(self):
        return cv2.arcLength(self.boundary[0], True)

    @property
    def roundness(self):
        return 4 * pi * (self.area / self.perimeter**2)

    @property
    def min_x(self):
        return np.min(self.boundary[0], axis=0)[0, 0]

    @property
    def max_x(self):
        return np.max(self.boundary[0], axis=0)[0, 0]


class Particle2D(MicroEntity2D):
    pass


class Cell2D(MicroEntity2D):
    def __init__(self, name, mask, x, y, custom_data=None):
        super().__init__(name, mask, x, y, custom_data)

    def dilated_mask(self, dilation_size=1):
        return morph_dilate(self.mask, "circle", half_kernel_size=dilation_size)

    def eroded_mask(self, dilation_size=1):
        return morph_erode(self.mask, "circle", half_kernel_size=dilation_size)

    @staticmethod
    def from_mask(cell_mask, cell_name, area_range=None, custom_data={}):
        mask = np.zeros(cell_mask.shape, dtype=np.uint8)
        cnts = contours_find(cell_mask)
        if len(cnts) == 1:
            cnt = cnts[0]
            if len(cnt) >= 4:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    return None
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                mask = contours_fill(mask, [cnt], color=255)

                if area_range is not None:
                    area = cv2.countNonZero(mask)
                    if area_range[0] <= area <= area_range[1]:
                        return Cell2D(cell_name, mask, cx, cy, custom_data=custom_data)
                    else:
                        return None
                else:
                    return Cell2D(cell_name, mask, cx, cy, custom_data=custom_data)
        return None


class Synapse(MicroEntity):
    def __init__(self, cell_1, cell_2, custom_data=None):
        super().__init__(f"{cell_1.name}-{cell_2.name}", custom_data)
        self.cell_1 = cell_1
        self.cell_2 = cell_2

    @property
    def angle(self):
        return self.cell_1.angle_with_x_axis(self.cell_2)

    @property
    def distance(self):
        return self.cell_1.distance(self.cell_2)

    def front_cell_1(self):
        line = [self.cell_1.as_int_tuple(), self.cell_2.as_int_tuple()]
        return intersection_with_line(self.cell_1.mask, line)

    def crop(self, padding=5):
        synapse_mask = self.cell_1.mask | self.cell_2.mask
        x, y, w, h = cv2.boundingRect(synapse_mask)
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = w + padding * 2
        h = h + padding * 2
        return x, y, w, h
