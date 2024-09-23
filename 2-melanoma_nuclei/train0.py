from numena.io.json import Serializable
from abc import ABC, abstractmethod
from typing import List
from builtins import print
from dataclasses import dataclass, field
from dataclasses import InitVar, dataclass, field
from numena.enums import IMAGE_UINT8_COLOR_1C
from numena.image.basics import image_new
from numena.image.basics import image_new, image_split
from numena.image.color import bgr2hed, bgr2hsv, gray2rgb, rgb2bgr
from numena.image.contour import contours_find
from numena.image.morphology import WatershedSkimage
from numena.image.threshold import threshold_tozero
from numena.io.drive import Directory
from numena.io.json import json_read, json_write
from typing import List
from typing import List, NewType
from typing import List, Tuple
import argparse
import ast
import copy
import cv2
import numpy as np
import os
import pandas as pd
import random
import simplejson
import time
from numena.image.drawing import (
    draw_overlay,
    fill_ellipses_as_labels,
    fill_polygons_as_labels,
)
from enum import Enum
from numena.image.basics import image_ew_max, image_ew_mean, image_ew_min
from numena.image.basics import image_split
from numena.image.color import bgr2hed, bgr2hsv, rgb2bgr, rgb2hed
from numena.image.morphology import morph_fill
from numena.image.threshold import threshold_binary, threshold_tozero
from numena.io.image import imread_color, imread_grayscale, imread_tiff
from numena.io.imagej import read_ellipses_from_csv, read_polygons_from_roi
from numena.time import eventid
from scipy.stats import kurtosis, skew
from skimage.morphology import remove_small_holes, remove_small_objects
from typing import List, NewType

from kartezio.endpoint import EndpointWatershed
from kartezio.dataset import read_dataset
from kartezio.training import train_model
from kartezio.apps.instance_segmentation import create_instance_segmentation_model

model = create_instance_segmentation_model(
    generations=10,
    _lambda=5,
    inputs=3,
    outputs=2,
    endpoint=EndpointWatershed(),
)
elite, _ = train_model(model, read_dataset("dataset"), ".")
