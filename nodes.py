from numena.enums import IMAGE_UINT8_POSITIVE
from numena.image.basics import image_ew_max
from numena.image.basics import image_ew_mean
from numena.image.basics import image_ew_min
from numena.image.color import rgb2bgr
from numena.image.morphology import morph_fill
from numena.image.threshold import threshold_binary
from scipy.stats import kurtosis
from scipy.stats import skew
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
import cv2
import numpy as np


def threshold_tozero(image, threshold):
    return cv2.threshold(image, threshold, IMAGE_UINT8_POSITIVE,
                         cv2.THRESH_TOZERO)[1]


def threshold_binary(image, threshold, value=IMAGE_UINT8_POSITIVE):
    return cv2.threshold(image, threshold, value, cv2.THRESH_BINARY)[1]


class Registry:

    class SubRegistry:

        def __init__(self):
            self.components = []

        def add(self, item_name, replace=False):

            def inner(item_cls):
                self.components.append(item_cls)

                def wrapper(*args, **kwargs):
                    return item_cls(*args, **kwargs)

                return wrapper

            return inner

    def __init__(self):
        self.nodes = self.SubRegistry()


registry = Registry()


class Node:

    def __init__(self, name, symbol, arity, args, sources):
        self.name = name
        self.symbol = symbol
        self.arity = arity
        self.args = args
        self.sources = sources


SHARPEN_KERNEL = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")
ROBERT_CROSS_H_KERNEL = np.array(([0, 1], [-1, 0]), dtype="int")
ROBERT_CROSS_V_KERNEL = np.array(([1, 0], [0, -1]), dtype="int")
OPENCV_MIN_KERNEL_SIZE = 3
OPENCV_MAX_KERNEL_SIZE = 31
OPENCV_KERNEL_RANGE = OPENCV_MAX_KERNEL_SIZE - OPENCV_MIN_KERNEL_SIZE
OPENCV_MIN_INTENSITY = 0
OPENCV_MAX_INTENSITY = 255
OPENCV_INTENSITY_RANGE = OPENCV_MAX_INTENSITY - OPENCV_MIN_INTENSITY
KERNEL_SCALE = OPENCV_KERNEL_RANGE / OPENCV_INTENSITY_RANGE
GABOR_SIGMAS = [
    0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
]
GABOR_THETAS = np.arange(0, 2, step=1.0 / 8) * np.pi
GABOR_LAMBDS = [
    0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
]
GABOR_GAMMAS = np.arange(0.0625, 1.001, step=1.0 / 16)


def clamp_ksize(ksize):
    if ksize < OPENCV_MIN_KERNEL_SIZE:
        return OPENCV_MIN_KERNEL_SIZE
    if ksize > OPENCV_MAX_KERNEL_SIZE:
        return OPENCV_MAX_KERNEL_SIZE
    return ksize


def remap_ksize(ksize):
    return int(round(ksize * KERNEL_SCALE + OPENCV_MIN_KERNEL_SIZE))


def unodd_ksize(ksize):
    if ksize % 2 == 0:
        return ksize + 1
    return ksize


def correct_ksize(ksize):
    ksize = remap_ksize(ksize)
    ksize = clamp_ksize(ksize)
    ksize = unodd_ksize(ksize)
    return ksize


def ellipse_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def cross_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))


def rect_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def gabor_kernel(ksize, p1, p2):
    ksize = clamp_ksize(ksize)
    ksize = unodd_ksize(ksize)
    p1_bin = "{0:08b}".format(p1)
    p2_bin = "{0:08b}".format(p2)
    sigma = GABOR_SIGMAS[int(p1_bin[:4], 2)]
    theta = GABOR_THETAS[int(p1_bin[4:], 2)]
    lambd = GABOR_LAMBDS[int(p2_bin[:4], 2)]
    gamma = GABOR_GAMMAS[int(p2_bin[4:], 2)]
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)


def kernel_from_parameters(p):
    # 50%
    if p[1] < 128:
        return ellipse_kernel(p[0])
    # 25%
    if p[1] < 192:
        return cross_kernel(p[0])
    # 25%
    return rect_kernel(p[0])


@registry.nodes.add("max")
class Max(Node):

    def __init__(self):
        super().__init__("max", "MAX", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_max(x[0], x[1])


@registry.nodes.add("min")
class Min(Node):

    def __init__(self):
        super().__init__("min", "MIN", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_min(x[0], x[1])


@registry.nodes.add("mean")
class Mean(Node):

    def __init__(self):
        super().__init__("mean", "MEAN", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_mean(x[0], x[1])


@registry.nodes.add("add")
class Add(Node):

    def __init__(self):
        super().__init__("add", "ADD", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.add(x[0], x[1])


@registry.nodes.add("subtract")
class Subtract(Node):

    def __init__(self):
        super().__init__("subtract", "SUB", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.subtract(x[0], x[1])


@registry.nodes.add("bitwise_not")
class BitwiseNot(Node):

    def __init__(self):
        super().__init__("bitwise_not", "NOT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_not(x[0])


@registry.nodes.add("bitwise_or")
class BitwiseOr(Node):

    def __init__(self):
        super().__init__("bitwise_or", "BOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_or(x[0], x[1])


@registry.nodes.add("bitwise_and")
class BitwiseAnd(Node):

    def __init__(self):
        super().__init__("bitwise_and", "BAND", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[1])


@registry.nodes.add("bitwise_and_mask")
class BitwiseAndMask(Node):

    def __init__(self):
        super().__init__("bitwise_and_mask", "ANDM", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[0], mask=x[1])


@registry.nodes.add("bitwise_xor")
class BitwiseXor(Node):

    def __init__(self):
        super().__init__("bitwise_xor", "BXOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_xor(x[0], x[1])


@registry.nodes.add("sqrt")
class SquareRoot(Node):

    def __init__(self):
        super().__init__("sqrt", "SQRT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.sqrt(
            (x[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)


@registry.nodes.add("pow2")
class Square(Node):

    def __init__(self):
        super().__init__("pow2", "POW", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.pow(
            (x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


@registry.nodes.add("exp")
class Exp(Node):

    def __init__(self):
        super().__init__("exp", "EXP", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.exp(
            (x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


@registry.nodes.add("log")
class Log(Node):

    def __init__(self):
        super().__init__("log", "LOG", 1, 0, sources="Numpy")

    def call(self, x, args=None):
        return np.log1p(x[0]).astype(np.uint8)


@registry.nodes.add("median_blur")
class MedianBlur(Node):

    def __init__(self):
        super().__init__("median_blur", "BLRM", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        return cv2.medianBlur(x[0], ksize)


@registry.nodes.add("gaussian_blur")
class GaussianBlur(Node):

    def __init__(self):
        super().__init__("gaussian_blur", "BLRG", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@registry.nodes.add("laplacian")
class Laplacian(Node):

    def __init__(self):
        super().__init__("laplacian", "LPLC", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.Laplacian(x[0], cv2.CV_64F).astype(np.uint8)


@registry.nodes.add("sobel")
class Sobel(Node):

    def __init__(self):
        super().__init__("sobel", "SOBL", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        if args[1] < 128:
            return cv2.Sobel(x[0], cv2.CV_64F, 1, 0,
                             ksize=ksize).astype(np.uint8)
        return cv2.Sobel(x[0], cv2.CV_64F, 0, 1, ksize=ksize).astype(np.uint8)


@registry.nodes.add("robert_cross")
class RobertCross(Node):

    def __init__(self):
        super().__init__("robert_cross", "RBRT", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        img = (x[0] / 255.0).astype(np.float32)
        h = cv2.filter2D(img, -1, ROBERT_CROSS_H_KERNEL)
        v = cv2.filter2D(img, -1, ROBERT_CROSS_V_KERNEL)
        return (cv2.sqrt(cv2.pow(h, 2) + cv2.pow(v, 2)) * 255).astype(np.uint8)


@registry.nodes.add("canny")
class Canny(Node):

    def __init__(self):
        super().__init__("canny", "CANY", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.Canny(x[0], args[0], args[1])


@registry.nodes.add("sharpen")
class Sharpen(Node):

    def __init__(self):
        super().__init__("sharpen", "SHRP", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.filter2D(x[0], -1, SHARPEN_KERNEL)


@registry.nodes.add("gabor")
class GaborFilter(Node):

    def __init__(self, ksize=11):
        super().__init__("gabor", "GABR", 1, 2, sources="OpenCV")
        self.ksize = ksize

    def call(self, x, args=None):
        gabor_k = gabor_kernel(self.ksize, args[0], args[1])
        return cv2.filter2D(x[0], -1, gabor_k)


@registry.nodes.add("abs_diff")
class AbsoluteDifference(Node):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("abs_diff", "ABSD", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        image = x[0].copy()
        return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + args[1]


@registry.nodes.add("abs_diff2")
class AbsoluteDifference2(Node):

    def __init__(self):
        super().__init__("abs_diff2", "ABS2", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return 255 - cv2.absdiff(x[0], x[1])


@registry.nodes.add("fluo_tophat")
class FluoTopHat(Node):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("fluo_tophat", "FLUO", 1, 2, sources="Handmade")

    def _rescale_intensity(self, img, min_val, max_val):
        output_img = np.clip(img, min_val, max_val)
        if max_val - min_val == 0:
            return (output_img * 255).astype(np.uint8)
        output_img = (output_img - min_val) / (max_val - min_val) * 255
        return output_img.astype(np.uint8)

    def call(self, x, args=None):
        kernel = kernel_from_parameters(args)
        img = cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel, iterations=10)
        kur = np.mean(kurtosis(img, fisher=True))
        skew1 = np.mean(skew(img))
        if kur > 1 and skew1 > 1:
            p2, p98 = np.percentile(img, (15, 99.5), method="linear")
        else:
            p2, p98 = np.percentile(img, (15, 100), method="linear")
        return self._rescale_intensity(img, p2, p98)


@registry.nodes.add("rel_diff")
class RelativeDifference(Node):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("rel_diff", "RELD", 1, 1, sources="Handmade")

    def call(self, x, args=None):
        img = x[0]
        max_img = np.max(img)
        min_img = np.min(img)
        ksize = correct_ksize(args[0])
        gb = cv2.GaussianBlur(img, (ksize, ksize), 0)
        gb = np.float32(gb)
        img = np.divide(img, gb + 1e-15, dtype=np.float32)
        img = cv2.normalize(img, img, max_img, min_img, cv2.NORM_MINMAX)
        return img.astype(np.uint8)


@registry.nodes.add("erode")
class Erode(Node):

    def __init__(self):
        super().__init__("erode", "EROD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.erode(inputs[0], kernel)


@registry.nodes.add("dilate")
class Dilate(Node):

    def __init__(self):
        super().__init__("dilate", "DILT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.dilate(inputs[0], kernel)


@registry.nodes.add("open")
class Open(Node):

    def __init__(self):
        super().__init__("open", "OPEN", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_OPEN, kernel)


@registry.nodes.add("close")
class Close(Node):

    def __init__(self):
        super().__init__("close", "CLSE", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_CLOSE, kernel)


@registry.nodes.add("morph_gradient")
class MorphGradient(Node):

    def __init__(self):
        super().__init__("morph_gradient", "MGRD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_GRADIENT, kernel)


@registry.nodes.add("morph_tophat")
class MorphTopHat(Node):

    def __init__(self):
        super().__init__("morph_tophat", "MTHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_TOPHAT, kernel)


@registry.nodes.add("morph_blackhat")
class MorphBlackHat(Node):

    def __init__(self):
        super().__init__("morph_blackhat", "MBHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_BLACKHAT, kernel)


@registry.nodes.add("fill_holes")
class FillHoles(Node):

    def __init__(self):
        super().__init__("fill_holes", "FILL", 1, 0, sources="Handmade")

    def call(self, inputs, p):
        return morph_fill(inputs[0])


@registry.nodes.add("remove_small_objects")
class RemoveSmallObjects(Node):

    def __init__(self):
        super().__init__("remove_small_objects",
                         "RMSO",
                         1,
                         1,
                         sources="Skimage")

    def call(self, x, args=None):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@registry.nodes.add("remove_small_holes")
class RemoveSmallHoles(Node):

    def __init__(self):
        super().__init__("remove_small_holes", "RMSH", 1, 1, sources="Skimage")

    def call(self, x, args=None):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@registry.nodes.add("threshold")
class Threshold(Node):

    def __init__(self):
        super().__init__("threshold", "TRH", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return threshold_binary(x[0], args[1])
        return threshold_tozero(x[0], args[1])


@registry.nodes.add("threshold_at_1")
class ThresholdAt1(Node):

    def __init__(self):
        super().__init__("threshold_at_1", "TRH1", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return threshold_binary(x[0], 1)
        return threshold_tozero(x[0], 1)


# @registry.nodes.add("TRHA")
class ThresholdAdaptive(Node):

    def __init__(self):
        super().__init__("adaptive_threshold", "TRHA", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        C = args[1] - 128  # to allow negative values
        return cv2.adaptiveThreshold(
            x[0],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ksize,
            C,
        )


@registry.nodes.add("distance_transform")
class DistanceTransform(Node):

    def __init__(self):
        super().__init__("distance_transform", "DTRF", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.normalize(
            cv2.distanceTransform(x[0].copy(), cv2.DIST_L2, 3),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )


@registry.nodes.add("distance_transform_and_thresh")
class DistanceTransformAndThresh(Node):

    def __init__(self):
        super().__init__("distance_transform_and_thresh",
                         "DTTR",
                         1,
                         2,
                         sources="OpenCV")

    def call(self, x, args=None):
        d = cv2.normalize(
            cv2.distanceTransform(x[0].copy(), cv2.DIST_L2, 3),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )
        return threshold_binary(d, args[0])


@registry.nodes.add("inrange_bin")
class BinaryInRange(Node):

    def __init__(self):
        super().__init__("inrange_bin", "BRNG", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.inRange(x[0], lower, upper)


@registry.nodes.add("inrange")
class InRange(Node):

    def __init__(self):
        super().__init__("inrange", "RNG", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.bitwise_and(
            x[0],
            x[0],
            mask=cv2.inRange(x[0], lower, upper),
        )