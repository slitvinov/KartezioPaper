from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from enum import Enum
from numba import jit
from numena.image.basics import image_ew_max
from numena.image.basics import image_ew_mean
from numena.image.basics import image_ew_min
from numena.image.basics import image_new
from numena.image.basics import image_split
from numena.image.color import rgb2bgr
from numena.image.drawing import fill_polygons_as_labels
from numena.image.morphology import morph_fill
from numena.image.morphology import WatershedSkimage
from numena.image.threshold import threshold_binary
from numena.image.threshold import threshold_tozero
from numena.io.image import imread_color
from numena.io.imagej import read_polygons_from_roi
from numena.io.json import json_read
from numena.io.json import json_write
from numena.time import eventid
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis
from scipy.stats import skew
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from typing import List
from typing import NewType
from typing import Tuple
import argparse
import copy
import cv2
import numpy as np
import os
import pandas as pd
import random
import simplejson
import time


@dataclass
class Directory:
    path: InitVar[str]
    _path: Path = field(init=False)

    def __post_init__(self, path):
        self._path = Path(path)

    def __getattr__(self, attr):
        return getattr(self._path, attr)

    def __truediv__(self, key):
        return self._path / key

    def read(self, filename):
        filepath = self / filename
        extension = filepath.suffix
        filepath = str(filepath)
        return pd.read_csv(filepath)

    def next(self, next_location):
        filepath = self / next_location
        filepath.mkdir(parents=True, exist_ok=True)
        return Directory(filepath)

    def ls(self, regex="*", ordered=False):
        return sorted(self.glob(regex))


class Registry:

    class SubRegistry:

        def __init__(self):
            self.__components = {}

        def add(self, item_name, replace=False):

            def inner(item_cls):
                self.__components[item_name] = item_cls

                def wrapper(*args, **kwargs):
                    return item_cls(*args, **kwargs)

                return wrapper

            return inner

        def get(self, item_name):
            return self.__components[item_name]

        def instantiate(self, item_name, *args, **kwargs):
            return self.get(item_name)(*args, **kwargs)

        def list(self):
            return self.__components

    def __init__(self):
        self.nodes = self.SubRegistry()
        self.stackers = self.SubRegistry()
        self.metrics = self.SubRegistry()
        self.readers = self.SubRegistry()


registry = Registry()
JSON_ELITE = "elite.json"
JSON_HISTORY = "history.json"
JSON_META = "META.json"
CSV_DATASET = "dataset.csv"
DIR_PREVIEW = "__preview__"


class Factory:
    """
    Using Factory Pattern:
    https://refactoring.guru/design-patterns/factory-method
    """

    def __init__(self, prototype):
        self._prototype = None
        self.set_prototype(prototype)

    def set_prototype(self, prototype):
        self._prototype = prototype

    def create(self):
        return self._prototype.clone()

class Observable:

    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self, event):
        for observer in self._observers:
            observer.update(event)


def pack_one_directory(directory_path):
    directory = Directory(directory_path)
    packed_history = {}
    elite = json_read(filepath=f"{directory_path}/elite.json")
    packed_history["dataset"] = elite["dataset"]
    packed_history["decoding"] = elite["decoding"]
    packed_history["elite"] = elite["individual"]
    packed_history["generations"] = []
    generations = []
    for g in directory.ls(f"G*.json", ordered=True):
        generations.append(int(g.name.replace("G", "").split(".")[0]))
    generations.sort()
    for generation in generations:
        current_generation = json_read(
            filepath=f"{directory_path}/G{generation}.json")
        generation_json = {
            "generation": generation,
            "population": current_generation["population"],
        }
        packed_history["generations"].append(generation_json)
    json_write(filepath=f"{directory_path}/history.json",
               json_data=packed_history,
               indent=None)
    print(f"All generations packed in {directory_path}.")
    for generation in generations:
        file_to_delete = f"{directory_path}/G{generation}.json"
        os.remove(file_to_delete)
    print(f"All {len(generations)} generation files deleted.")


def from_individual(individual):
    return {
        "sequence": simplejson.dumps(individual.sequence.tolist()),
        "fitness": individual.fitness,
    }


def from_population(population):
    json_data = []
    for individual_idx, individual in population:
        json_data.append(from_individual(individual))
    return json_data


def from_dataset(dataset):
    return {
        "name": dataset.name,
        "label_name": dataset.label_name,
        "indices": dataset.indices,
    }


class JsonSaver:

    def __init__(self, dataset, parser):
        self.dataset_json = from_dataset(dataset)
        self.parser_as_json = parser.dumps()

    def save_population(self, filepath, population):
        json_data = {
            "dataset": self.dataset_json,
            "population": from_population(population),
            "decoding": self.parser_as_json,
        }
        json_write(filepath, json_data)

    def save_individual(self, filepath, individual):
        json_data = {
            "dataset": self.dataset_json,
            "individual": from_individual(individual),
            "decoding": self.parser_as_json,
        }
        json_write(filepath, json_data)


class KartezioNode:

    def __init__(self,
                 name: str,
                 symbol: str,
                 arity: int,
                 args: int,
                 sources=None):
        self.name = name
        self.symbol = symbol
        self.arity = arity
        self.args = args
        self.sources = sources

    def dumps(self) -> dict:
        return {
            "name": self.name,
            "abbv": self.symbol,
            "arity": self.arity,
            "args": self.args,
            "kwargs": self._to_json_kwargs(),
        }

class KartezioStacker(KartezioNode):
    def __init__(self, name: str, symbol: str, arity: int):
        super().__init__(name, symbol, arity, 0)



@registry.stackers.add("MEAN")
class StackerMean(KartezioStacker):
    def __init__(self,
                 name="mean_stacker",
                 symbol="MEAN",
                 arity=1,
                 threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y):
        return np.mean(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        return threshold_tozero(yi, self.threshold)


@registry.stackers.add("SUM")
class StackerSum(KartezioStacker):

    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self,
                 name="Sum KartezioStacker",
                 symbol="SUM",
                 arity=1,
                 threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y):
        stack_array = np.array(Y).astype(np.float32)
        stack_array /= 255.0
        stack_sum = np.sum(stack_array, axis=0)
        return stack_sum.astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 0:
            return cv2.GaussianBlur(yi, (7, 7), 1)
        return yi


@registry.stackers.add("MIN")
class StackerMin(KartezioStacker):

    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self, name="min_stacker", symbol="MIN", arity=1, threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y):
        return np.min(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        return threshold_tozero(yi, self.threshold)


@registry.stackers.add("MAX")
class StackerMax(KartezioStacker):

    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self, name="max_stacker", symbol="MAX", arity=1, threshold=1):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y):
        return np.max(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 0:
            return cv2.GaussianBlur(yi, (7, 7), 1)
        return yi

class KartezioEndpoint(KartezioNode):
    def __init__(self, name: str, symbol: str, arity: int, outputs_keys: list):
        super().__init__(name, symbol, arity, 0)
        self.outputs_keys = outputs_keys


class KartezioBundle:

    def __init__(self):
        self.__nodes = {}
        self.fill()


    def add_node(self, node_name):
        self.__nodes[len(self.__nodes)] = registry.nodes.instantiate(node_name)

    def arity_of(self, i):
        return self.__nodes[i].arity

    def execute(self, name, x, args):
        return self.__nodes[name].call(x, args)

    @property
    def size(self):
        return len(self.__nodes)

    @property
    def ordered_list(self):
        return [self.__nodes[i].name for i in range(self.size)]


class KartezioGenome:
    def __init__(self, shape: tuple = (14, 5), sequence: np.ndarray = None):
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = np.zeros(shape=shape, dtype=np.uint8)

    def __deepcopy__(self, memo={}):
        new = self.__class__(*self.sequence.shape)
        new.sequence = self.sequence.copy()
        return new

    def __getitem__(self, item):
        return self.sequence.__getitem__(item)

    def __setitem__(self, key, value):
        return self.sequence.__setitem__(key, value)

    def clone(self):
        return copy.deepcopy(self)

class GenomeFactory(Factory):

    def __init__(self, prototype):
        super().__init__(prototype)


class GenomeAdapter:
    pass


class GenomeWriter(GenomeAdapter):

    def write_function(self, genome, node, function_id):
        genome[g.inputs + node, 0] = function_id

    def write_connections(self, genome, node, connections):
        genome[g.inputs + node,
               1:g.para_idx] = connections

    def write_parameters(self, genome, node, parameters):
        genome[g.inputs + node, g.para_idx:] = parameters

    def write_output_connection(self, genome, output_index, connection):
        genome[g.out_idx + output_index,
               1] = connection


class GenomeReader(GenomeAdapter):

    def read_function(self, genome, node):
        return genome[g.inputs + node, 0]

    def read_connections(self, genome, node):
        return genome[g.inputs + node,
                      1:g.para_idx]

    def read_active_connections(self, genome, node, active_connections):
        return genome[
            g.inputs + node,
            1:1 + active_connections,
        ]

    def read_parameters(self, genome, node):
        return genome[g.inputs + node, g.para_idx:]

    def read_outputs(self, genome):
        return genome[g.out_idx:, :]


class GenomeReaderWriter(GenomeReader, GenomeWriter):
    pass



class KartezioParser(GenomeReader):
    def __init__(self):
        super().__init__()

    def dumps(self) -> dict:
        return {
            "metadata": {
                "rows": 1,
                "columns": g.nodes,
                "n_in": g.inputs,
                "n_out": g.outputs,
                "n_para": g.parameters,
                "n_conn": g.arity,
            },
            "functions": g.bundle.ordered_list,
            "endpoint": g.endpoint.dumps(),
            "mode": "default",
        }

    def _parse_one_graph(self, genome, graph_source):
        next_indices = graph_source.copy()
        output_tree = graph_source.copy()
        while next_indices:
            next_index = next_indices.pop()
            if next_index < g.inputs:
                continue
            function_index = self.read_function(genome,
                                                next_index - g.inputs)
            active_connections = g.bundle.arity_of(function_index)
            next_connections = set(
                self.read_active_connections(genome,
                                             next_index - g.inputs,
                                             active_connections))
            next_indices = next_indices.union(next_connections)
            output_tree = output_tree.union(next_connections)
        return sorted(list(output_tree))

    def parse_to_graphs(self, genome):
        outputs = self.read_outputs(genome)
        graphs_list = [
            self._parse_one_graph(genome, {output[1]})
            for output in outputs
        ]
        return graphs_list

    def _x_to_output_map(self, genome: KartezioGenome, graphs_list, x):
        output_map = {i: x[i].copy() for i in range(g.inputs)}
        for graph in graphs_list:
            for node in graph:
                if node < g.inputs:
                    continue
                node_index = node - g.inputs
                function_index = self.read_function(genome, node_index)
                arity = g.bundle.arity_of(function_index)
                connections = self.read_active_connections(
                    genome, node_index, arity)
                inputs = [output_map[c] for c in connections]
                p = self.read_parameters(genome, node_index)
                value = g.bundle.execute(function_index, inputs, p)
                output_map[node] = value
        return output_map

    def _parse_one(self, genome, graphs_list, x):
        output_map = self._x_to_output_map(genome, graphs_list, x)
        return [
            output_map[output_gene[1]]
            for output_gene in self.read_outputs(genome)
        ]

    def parse_population(self, population, x):
        y_pred = []
        for i in range(len(population.individuals)):
            y, t = self.parse(population.individuals[i], x)
            population.set_time(i, t)
            y_pred.append(y)
        return y_pred

    def parse(self, genome, x):
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genome)
        # for each image
        for xi in x:
            start_time = time.time()
            y_pred = self._parse_one(genome, graphs, xi)
            y_pred = g.endpoint.call(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time


class ExportableNode(KartezioNode):
    pass

class KartezioCallback:

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.parser = None

    def set_parser(self, parser):
        self.parser = parser

    def update(self, event):
        if event["n"] % self.frequency == 0 or event["force"]:
            self._callback(event["n"], event["name"], event["content"])

class KartezioMetric(KartezioNode):

    def __init__(
        self,
        name: str,
        symbol: str,
        arity: int,
    ):
        super().__init__(name, symbol, arity, 0)


class KartezioFitness(KartezioNode):

    def __init__(
        self,
        name: str,
        symbol: str,
        arity: int,
        default_metric: KartezioMetric = None,
    ):
        super().__init__(name, symbol, arity, 0)
        self.metrics = []
        self.add_metric(default_metric)

    def add_metric(self, metric: KartezioMetric):
        self.metrics.append(metric)

    def call(self, y_true, y_pred):
        scores = []
        for yi_pred in y_pred:
            scores.append(self.compute_one(y_true, yi_pred))
        return scores

    def compute_one(self, y_true, y_pred):
        score = 0.0
        y_size = len(y_true)
        for i in range(y_size):
            _y_true = y_true[i].copy()
            _y_pred = y_pred[i]
            score += self.__fitness_sum(_y_true, _y_pred)
        return score / y_size

    def __fitness_sum(self, y_true, y_pred):
        score = 0.0
        for metric in self.metrics:
            score += metric.call(y_true, y_pred)
        return score

class KartezioMutation(GenomeReaderWriter):

    def __init__(self, n_functions):
        super().__init__()
        self.n_functions = n_functions
        self.parameter_max_value = 256

    @property
    def random_parameters(self):
        return np.random.randint(self.parameter_max_value,
                                 size=g.parameters)

    @property
    def random_functions(self):
        return np.random.randint(self.n_functions)

    @property
    def random_output(self):
        return np.random.randint(g.out_idx, size=1)

    def random_connections(self, idx: int):
        return np.random.randint(g.inputs + idx,
                                 size=g.arity)

    def mutate_function(self, genome: KartezioGenome, idx: int):
        self.write_function(genome, idx, self.random_functions)

    def mutate_connections(self,
                           genome: KartezioGenome,
                           idx: int,
                           only_one: int = None):
        new_connections = self.random_connections(idx)
        new_value = new_connections[only_one]
        new_connections = self.read_connections(genome, idx)
        new_connections[only_one] = new_value
        self.write_connections(genome, idx, new_connections)

    def mutate_parameters(self,
                          genome,
                          idx,
                          only_one = None):
        new_parameters = self.random_parameters
        if only_one is not None:
            old_parameters = self.read_parameters(genome, idx)
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.write_parameters(genome, idx, new_parameters)

    def mutate_output(self, genome, idx):
        self.write_output_connection(genome, idx, self.random_output)


class KartezioPopulation:

    def __init__(self):
        self.individuals = [None] * (g._lambda + 1)
        self._fitness = {
            "fitness": np.zeros(g._lambda + 1),
            "time": np.zeros(g._lambda + 1)
        }

    def get_best_individual(self):
        pass

    def __getitem__(self, item):
        return self.individuals.__getitem__(item)

    def __setitem__(self, key, value):
        self.individuals.__setitem__(key, value)

    def set_time(self, individual, value):
        self._fitness["time"][individual] = value

    def set_fitness(self, fitness):
        self._fitness["fitness"] = fitness

    @property
    def fitness(self):
        return self._fitness["fitness"]

    @property
    def time(self):
        return self._fitness["time"]

    @property
    def score(self):
        score_list = list(zip(self.fitness, self.time))
        return np.array(score_list,
                        dtype=[("fitness", float), ("time", float)])


@jit(nopython=True)
def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


@registry.metrics.add("CAP")
class MetricCellpose(KartezioMetric):

    def __init__(self, thresholds=0.5):
        super().__init__("Cellpose Average Precision", symbol="CAP", arity=1)
        self.thresholds = thresholds
        if not isinstance(self.thresholds, list) and not isinstance(
                self.thresholds, np.ndarray):
            self.thresholds = [self.thresholds]
        self.n_thresholds = len(self.thresholds)

    def call(self, y_true: np.ndarray, y_pred: np.ndarray):
        _y_true = y_true[0]
        _y_pred = y_pred["labels"]
        ap, tp, fp, fn = self.average_precision(_y_true, _y_pred)
        return 1.0 - ap[0]

    def average_precision(self, masks_true, masks_pred):
        not_list = False
        if not isinstance(masks_true, list):
            masks_true = [masks_true]
            masks_pred = [masks_pred]
            not_list = True
        ap = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        tp = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        fp = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        fn = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        n_true = np.array(list(map(np.max, masks_true)))
        n_pred = np.array(list(map(np.max, masks_pred)))
        for n in range(len(masks_true)):
            #  _,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
            if n_pred[n] > 0:
                iou = _intersection_over_union(masks_true[n],
                                               masks_pred[n])[1:, 1:]
                for k, th in enumerate(self.thresholds):
                    tp[n, k] = self._true_positive(iou, th)
            fp[n] = n_pred[n] - tp[n]
            fn[n] = n_true[n] - tp[n]
            if tp[n] == 0:
                if n_true[n] == 0:
                    ap[n] = 1.0
                else:
                    ap[n] = 0.0
            else:
                ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        if not_list:
            ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
        return ap, tp, fp, fn

    def _true_positive(self, iou, th):
        n_min = min(iou.shape[0], iou.shape[1])
        costs = -(iou >= th).astype(float) - iou / (2 * n_min)
        true_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = iou[true_ind, pred_ind] >= th
        tp = match_ok.sum()
        return tp


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


class NodeImageProcessing(KartezioNode):

    def _to_json_kwargs(self) -> dict:
        return {}


@registry.nodes.add("max")
class Max(NodeImageProcessing):

    def __init__(self):
        super().__init__("max", "MAX", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_max(x[0], x[1])


@registry.nodes.add("min")
class Min(NodeImageProcessing):

    def __init__(self):
        super().__init__("min", "MIN", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_min(x[0], x[1])


@registry.nodes.add("mean")
class Mean(NodeImageProcessing):

    def __init__(self):
        super().__init__("mean", "MEAN", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_mean(x[0], x[1])


@registry.nodes.add("add")
class Add(ExportableNode):

    def __init__(self):
        super().__init__("add", "ADD", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.add(x[0], x[1])


@registry.nodes.add("subtract")
class Subtract(ExportableNode):

    def __init__(self):
        super().__init__("subtract", "SUB", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.subtract(x[0], x[1])


@registry.nodes.add("bitwise_not")
class BitwiseNot(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_not", "NOT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_not(x[0])

@registry.nodes.add("bitwise_or")
class BitwiseOr(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_or", "BOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_or(x[0], x[1])


@registry.nodes.add("bitwise_and")
class BitwiseAnd(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_and", "BAND", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[1])

@registry.nodes.add("bitwise_and_mask")
class BitwiseAndMask(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_and_mask", "ANDM", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[0], mask=x[1])


@registry.nodes.add("bitwise_xor")
class BitwiseXor(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_xor", "BXOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_xor(x[0], x[1])


@registry.nodes.add("sqrt")
class SquareRoot(NodeImageProcessing):

    def __init__(self):
        super().__init__("sqrt", "SQRT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.sqrt(
            (x[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)


@registry.nodes.add("pow2")
class Square(NodeImageProcessing):

    def __init__(self):
        super().__init__("pow2", "POW", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.pow(
            (x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


@registry.nodes.add("exp")
class Exp(NodeImageProcessing):

    def __init__(self):
        super().__init__("exp", "EXP", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.exp(
            (x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


@registry.nodes.add("log")
class Log(NodeImageProcessing):

    def __init__(self):
        super().__init__("log", "LOG", 1, 0, sources="Numpy")

    def call(self, x, args=None):
        return np.log1p(x[0]).astype(np.uint8)


@registry.nodes.add("median_blur")
class MedianBlur(NodeImageProcessing):

    def __init__(self):
        super().__init__("median_blur", "BLRM", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        return cv2.medianBlur(x[0], ksize)


@registry.nodes.add("gaussian_blur")
class GaussianBlur(NodeImageProcessing):

    def __init__(self):
        super().__init__("gaussian_blur", "BLRG", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@registry.nodes.add("laplacian")
class Laplacian(NodeImageProcessing):

    def __init__(self):
        super().__init__("laplacian", "LPLC", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.Laplacian(x[0], cv2.CV_64F).astype(np.uint8)


@registry.nodes.add("sobel")
class Sobel(NodeImageProcessing):

    def __init__(self):
        super().__init__("sobel", "SOBL", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        if args[1] < 128:
            return cv2.Sobel(x[0], cv2.CV_64F, 1, 0,
                             ksize=ksize).astype(np.uint8)
        return cv2.Sobel(x[0], cv2.CV_64F, 0, 1, ksize=ksize).astype(np.uint8)


@registry.nodes.add("robert_cross")
class RobertCross(NodeImageProcessing):

    def __init__(self):
        super().__init__("robert_cross", "RBRT", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        img = (x[0] / 255.0).astype(np.float32)
        h = cv2.filter2D(img, -1, ROBERT_CROSS_H_KERNEL)
        v = cv2.filter2D(img, -1, ROBERT_CROSS_V_KERNEL)
        return (cv2.sqrt(cv2.pow(h, 2) + cv2.pow(v, 2)) * 255).astype(np.uint8)


@registry.nodes.add("canny")
class Canny(NodeImageProcessing):

    def __init__(self):
        super().__init__("canny", "CANY", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.Canny(x[0], args[0], args[1])


@registry.nodes.add("sharpen")
class Sharpen(NodeImageProcessing):

    def __init__(self):
        super().__init__("sharpen", "SHRP", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.filter2D(x[0], -1, SHARPEN_KERNEL)


@registry.nodes.add("gabor")
class GaborFilter(NodeImageProcessing):

    def __init__(self, ksize=11):
        super().__init__("gabor", "GABR", 1, 2, sources="OpenCV")
        self.ksize = ksize

    def call(self, x, args=None):
        gabor_k = gabor_kernel(self.ksize, args[0], args[1])
        return cv2.filter2D(x[0], -1, gabor_k)


@registry.nodes.add("abs_diff")
class AbsoluteDifference(NodeImageProcessing):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("abs_diff", "ABSD", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        image = x[0].copy()
        return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + args[1]


@registry.nodes.add("abs_diff2")
class AbsoluteDifference2(NodeImageProcessing):

    def __init__(self):
        super().__init__("abs_diff2", "ABS2", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return 255 - cv2.absdiff(x[0], x[1])


@registry.nodes.add("fluo_tophat")
class FluoTopHat(NodeImageProcessing):
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
            p2, p98 = np.percentile(img, (15, 99.5), interpolation="linear")
        else:
            p2, p98 = np.percentile(img, (15, 100), interpolation="linear")
        return self._rescale_intensity(img, p2, p98)


@registry.nodes.add("rel_diff")
class RelativeDifference(NodeImageProcessing):
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
class Erode(ExportableNode):

    def __init__(self):
        super().__init__("erode", "EROD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.erode(inputs[0], kernel)


@registry.nodes.add("dilate")
class Dilate(ExportableNode):

    def __init__(self):
        super().__init__("dilate", "DILT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.dilate(inputs[0], kernel)


@registry.nodes.add("open")
class Open(ExportableNode):

    def __init__(self):
        super().__init__("open", "OPEN", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_OPEN, kernel)

@registry.nodes.add("close")
class Close(ExportableNode):

    def __init__(self):
        super().__init__("close", "CLSE", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_CLOSE, kernel)

@registry.nodes.add("morph_gradient")
class MorphGradient(ExportableNode):

    def __init__(self):
        super().__init__("morph_gradient", "MGRD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_GRADIENT, kernel)


@registry.nodes.add("morph_tophat")
class MorphTopHat(ExportableNode):

    def __init__(self):
        super().__init__("morph_tophat", "MTHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_TOPHAT, kernel)


@registry.nodes.add("morph_blackhat")
class MorphBlackHat(ExportableNode):

    def __init__(self):
        super().__init__("morph_blackhat", "MBHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_BLACKHAT, kernel)


@registry.nodes.add("fill_holes")
class FillHoles(ExportableNode):

    def __init__(self):
        super().__init__("fill_holes", "FILL", 1, 0, sources="Handmade")

    def call(self, inputs, p):
        return morph_fill(inputs[0])


@registry.nodes.add("remove_small_objects")
class RemoveSmallObjects(NodeImageProcessing):

    def __init__(self):
        super().__init__("remove_small_objects",
                         "RMSO",
                         1,
                         1,
                         sources="Skimage")

    def call(self, x, args=None):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@registry.nodes.add("remove_small_holes")
class RemoveSmallHoles(NodeImageProcessing):

    def __init__(self):
        super().__init__("remove_small_holes", "RMSH", 1, 1, sources="Skimage")

    def call(self, x, args=None):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@registry.nodes.add("threshold")
class Threshold(NodeImageProcessing):

    def __init__(self):
        super().__init__("threshold", "TRH", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return threshold_binary(x[0], args[1])
        return threshold_tozero(x[0], args[1])


@registry.nodes.add("threshold_at_1")
class ThresholdAt1(NodeImageProcessing):

    def __init__(self):
        super().__init__("threshold_at_1", "TRH1", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return threshold_binary(x[0], 1)
        return threshold_tozero(x[0], 1)


# @registry.nodes.add("TRHA")
class ThresholdAdaptive(NodeImageProcessing):

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
class DistanceTransform(NodeImageProcessing):

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
class DistanceTransformAndThresh(NodeImageProcessing):

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
class BinaryInRange(NodeImageProcessing):

    def __init__(self):
        super().__init__("inrange_bin", "BRNG", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.inRange(x[0], lower, upper)


@registry.nodes.add("inrange")
class InRange(NodeImageProcessing):

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


IMAGE_NODES_ABBV_LIST = registry.nodes.list().keys()

class BundleOpenCV(KartezioBundle):
    def fill(self):
        for node_abbv in IMAGE_NODES_ABBV_LIST:
            self.add_node(node_abbv)


class EndpointWatershed(KartezioEndpoint):

    def __init__(self, use_dt=False, markers_distance=21, markers_area=None):
        super().__init__("Marker-Based Watershed", "WSHD", 2, [])
        self.wt = WatershedSkimage(use_dt=use_dt,
                                   markers_distance=markers_distance,
                                   markers_area=markers_area)

    def call(self, x, args=None):
        mask = x[0]
        markers = x[1]
        mask, markers, labels = self.wt.apply(mask,
                                              markers=markers,
                                              mask=mask > 0)
        return {
            "mask_raw": x[0],
            "markers_raw": x[1],
            "mask": mask,
            "markers": markers,
            "count": len(np.unique(labels)) - 1,
            "labels": labels,
        }

    def _to_json_kwargs(self) -> dict:
        return {
            "use_dt": self.wt.use_dt,
            "markers_distance": self.wt.markers_distance,
            "markers_area": self.wt.markers_area,
        }


class Event(Enum):
    START_STEP = "on_step_start"
    END_STEP = "on_step_end"
    START_LOOP = "on_loop_start"
    END_LOOP = "on_loop_end"


class CallbackVerbose(KartezioCallback):

    def _callback(self, n, e_name, e_content):
        fitness, time = e_content.get_best_fitness()
        if e_name == Event.END_STEP:
            verbose = f"[G {n:04}] {fitness:.16f}"
            print(verbose)
        elif e_name == Event.END_LOOP:
            verbose = f"[G {n:04}] {fitness:.16f}, loop done."
            print(verbose)


class CallbackSave(KartezioCallback):

    def __init__(self, workdir, dataset, frequency=1):
        super().__init__(frequency)
        self.workdir = Directory(workdir).next(eventid())
        self.dataset = dataset
        self.json_saver = None

    def set_parser(self, parser):
        super().set_parser(parser)
        self.json_saver = JsonSaver(self.dataset, self.parser)

    def save_population(self, population, n):
        filename = f"G{n}.json"
        filepath = self.workdir / filename
        self.json_saver.save_population(filepath, population)

    def save_elite(self, elite):
        filepath = self.workdir / JSON_ELITE
        self.json_saver.save_individual(filepath, elite)

    def _callback(self, n, e_name, e_content):
        if e_name == Event.END_STEP or e_name == Event.END_LOOP:
            self.save_population(e_content.get_individuals(), n)
            self.save_elite(e_content.individuals[0])


class FitnessAP(KartezioFitness):
    def __init__(self, thresholds=0.5):
        super().__init__(
            name=f"Average Precision ({thresholds})",
            symbol="AP",
            arity=1,
            default_metric=registry.metrics.instantiate("CAP",
                                                        thresholds=thresholds),
        )


class GoldmanWrapper(KartezioMutation):

    def __init__(self, mutation, decoder):
        super().__init__(None)
        self.mutation = mutation
        self.parser = decoder

    def mutate(self, genome):
        changed = False
        active_nodes = self.parser.parse_to_graphs(genome)
        while not changed:
            genome = self.mutation.mutate(genome)
            new_active_nodes = self.parser.parse_to_graphs(genome)
            changed = active_nodes != new_active_nodes
        return genome


class MutationClassic(KartezioMutation):

    def __init__(self, n_functions, mutation_rate,
                 output_mutation_rate):
        super().__init__(n_functions)
        self.mutation_rate = mutation_rate
        self.output_mutation_rate = output_mutation_rate
        self.n_mutations = int(
            np.floor(g.nodes * g.w * self.mutation_rate))
        self.all_indices = np.indices((g.nodes, g.w))
        self.all_indices = np.vstack(
            (self.all_indices[0].ravel(), self.all_indices[1].ravel())).T
        self.sampling_range = range(len(self.all_indices))

    def mutate(self, genome):
        sampling_indices = np.random.choice(self.sampling_range,
                                            self.n_mutations,
                                            replace=False)
        sampling_indices = self.all_indices[sampling_indices]
        for idx, mutation_parameter_index in sampling_indices:
            if mutation_parameter_index == 0:
                self.mutate_function(genome, idx)
            elif mutation_parameter_index <= g.arity:
                connection_idx = mutation_parameter_index - 1
                self.mutate_connections(genome, idx, only_one=connection_idx)
            else:
                parameter_idx = mutation_parameter_index - g.arity - 1
                self.mutate_parameters(genome, idx, only_one=parameter_idx)
        for output in range(g.outputs):
            if random.random() < self.output_mutation_rate:
                self.mutate_output(genome, output)
        return genome


class MutationAllRandom(KartezioMutation):
    def __init__(self, n_functions: int):
        super().__init__(n_functions)

    def mutate(self, genome: KartezioGenome):
        # mutate genes
        for i in range(g.nodes):
            self.mutate_function(genome, i)
            self.mutate_connections(genome, i)
            self.mutate_parameters(genome, i)
        # mutate outputs
        for i in range(g.outputs):
            self.mutate_output(genome, i)
        return genome


class ModelGA:

    def __init__(self, strategy):
        self.strategy = strategy
        self.current_generation = 0

    def fit(self, x, y):
        pass

    def initialization(self):
        self.strategy.initialization()

    def is_satisfying(self):
        end_of_generations = self.current_generation >= g.generations
        best_fitness_reached = self.strategy.population.fitness[0] == 0.0
        return end_of_generations or best_fitness_reached

    def selection(self):
        self.strategy.selection()

    def reproduction(self):
        self.strategy.reproduction()

    def mutation(self):
        self.strategy.mutation()

    def evaluation(self, y_true, y_pred):
        self.strategy.evaluation(y_true, y_pred)

    def next(self):
        self.current_generation += 1


class ModelCGP(Observable):

    def __init__(self, strategy, parser):
        super().__init__()
        self.strategy = strategy
        self.parser = parser
        self.callbacks = []

    def fit(
        self,
        x,
        y,
    ):
        genetic_algorithm = ModelGA(self.strategy)
        genetic_algorithm.initialization()
        y_pred = self.parser.parse_population(self.strategy.population, x)
        genetic_algorithm.evaluation(y, y_pred)
        self._notify(0, Event.START_LOOP, force=True)
        while not genetic_algorithm.is_satisfying():
            self._notify(genetic_algorithm.current_generation,
                         Event.START_STEP)
            genetic_algorithm.selection()
            genetic_algorithm.reproduction()
            genetic_algorithm.mutation()
            y_pred = self.parser.parse_population(self.strategy.population, x)
            genetic_algorithm.evaluation(y, y_pred)
            genetic_algorithm.next()
            self._notify(genetic_algorithm.current_generation, Event.END_STEP)
        self._notify(genetic_algorithm.current_generation,
                     Event.END_LOOP,
                     force=True)
        history = self.strategy.population.history()
        elite = self.strategy.elite
        return elite, history

    def _notify(self, n, name, force=False):
        event = {
            "n": n,
            "name": name,
            "content": self.strategy.population.history(),
            "force": force,
        }
        self.notify(event)


class Dataset:

    class SubSet:

        def __init__(self, dataframe):
            self.x = []
            self.y = []
            self.v = []
            self.dataframe = dataframe

        def add_item(self, x, y):
            self.x.append(x)
            self.y.append(y)

        def add_visual(self, visual):
            self.v.append(visual)

        @property
        def xy(self):
            return self.x, self.y

        @property
        def xyv(self):
            return self.x, self.y, self.v

    def __init__(self,
                 train_set,
                 test_set,
                 name,
                 label_name,
                 inputs,
                 indices=None):
        self.train_set = train_set
        self.test_set = test_set
        self.name = name
        self.label_name = label_name
        self.inputs = inputs
        self.indices = indices

    @property
    def train_x(self):
        return self.train_set.x

    @property
    def train_y(self):
        return self.train_set.y

    @property
    def train_v(self):
        return self.train_set.v

    @property
    def test_x(self):
        return self.test_set.x

    @property
    def test_y(self):
        return self.test_set.y

    @property
    def test_v(self):
        return self.test_set.v

    @property
    def train_xy(self):
        return self.train_set.xy

    @property
    def test_xy(self):
        return self.test_set.xy

    @property
    def train_xyv(self):
        return self.train_set.xyv

    @property
    def test_xyv(self):
        return self.test_set.xyv

    @property
    def split(self):
        return self.train_x, self.train_y, self.test_x, self.test_y


class DatasetMeta:

    @staticmethod
    def write(
        filepath,
        name,
        input_type,
        input_format,
        label_type,
        label_format,
        label_name,
        scale=1.0,
        mode="dataframe",
        meta_filename=JSON_META,
    ):
        json_data = {
            "name": name,
            "scale": scale,
            "label_name": label_name,
            "mode": mode,
            "input": {
                "type": input_type,
                "format": input_format
            },
            "label": {
                "type": label_type,
                "format": label_format
            },
        }
        json_write(filepath + "/" + meta_filename, json_data)

    @staticmethod
    def read(filepath, meta_filename):
        return json_read(filepath / meta_filename)


class DataReader:

    def __init__(self, directory, scale=1.0):
        self.scale = scale
        self.directory = directory

    def read(self, filename, shape=None):
        if str(filename) == "nan":
            filepath = ""
        else:
            filepath = str(self.directory / filename)
        return self._read(filepath, shape)

    def _read(self, filepath, shape=None):
        pass


@dataclass
class DataItem:
    datalist: List
    shape: Tuple
    count: int
    visual: np.ndarray = None

    @property
    def size(self):
        return len(self.datalist)


@registry.readers.add("image_rgb")
class ImageRGBReader(DataReader):

    def _read(self, filepath, shape=None):
        image = imread_color(filepath, rgb=False)
        return DataItem(image_split(image),
                        image.shape[:2],
                        None,
                        visual=rgb2bgr(image))


@registry.readers.add("roi_polygon")
class RoiPolygonReader(DataReader):

    def _read(self, filepath, shape=None):
        label_mask = image_new(shape)
        if filepath == "":
            return DataItem([label_mask], shape, 0)
        polygons = read_polygons_from_roi(filepath)
        fill_polygons_as_labels(label_mask, polygons)
        return DataItem([label_mask], shape, len(polygons))


@dataclass
class DatasetReader(Directory):
    counting: bool = False
    preview_dir: Directory = field(init=False)

    def __post_init__(self, path):
        super().__post_init__(path)

    def _read_meta(self, meta_filename):
        meta = DatasetMeta.read(self._path, meta_filename=meta_filename)
        self.name = meta["name"]
        self.scale = meta["scale"]
        self.mode = meta["mode"]
        self.label_name = meta["label_name"]
        input_reader_name = f"{meta['input']['type']}_{meta['input']['format']}"
        label_reader_name = f"{meta['label']['type']}_{meta['label']['format']}"
        self.input_reader = registry.readers.instantiate(input_reader_name,
                                                         directory=self,
                                                         scale=self.scale)
        self.label_reader = registry.readers.instantiate(label_reader_name,
                                                         directory=self,
                                                         scale=self.scale)

    def read_dataset(self,
                     dataset_filename=CSV_DATASET,
                     meta_filename=JSON_META,
                     indices=None):
        self._read_meta(meta_filename)
        if self.mode == "dataframe":
            return self._read_from_dataframe(dataset_filename, indices)
        raise AttributeError(f"{self.mode} is not handled yet")

    def _read_from_dataframe(self, dataset_filename, indices):
        dataframe = self.read(dataset_filename)
        dataframe_training = dataframe[dataframe["set"] == "training"]
        training = self._read_dataset(dataframe_training, indices)
        dataframe_testing = dataframe[dataframe["set"] == "testing"]
        testing = self._read_dataset(dataframe_testing)
        input_sizes = []
        [input_sizes.append(len(xi)) for xi in training.x]
        [input_sizes.append(len(xi)) for xi in testing.x]
        input_sizes = np.array(input_sizes)
        inputs = int(input_sizes[0])
        return Dataset(training, testing, self.name, self.label_name, inputs,
                       indices)

    def _read_dataset(self, dataframe, indices=None):
        dataset = Dataset.SubSet(dataframe)
        dataframe.reset_index(inplace=True)
        for row in dataframe.itertuples():
            x = self.input_reader.read(row.input, shape=None)
            y = self.label_reader.read(row.label, shape=x.shape)
            y = y.datalist
            dataset.n_inputs = x.size
            dataset.add_item(x.datalist, y)
            visual_from_table = False
            if not visual_from_table:
                dataset.add_visual(x.visual)
        return dataset


class IndividualHistory:

    def __init__(self):
        self.fitness = {"fitness": 0.0, "time": 0.0}
        self.sequence = None

    def set_values(self, sequence, fitness, time):
        self.sequence = sequence
        self.fitness["fitness"] = fitness
        self.fitness["time"] = time


class PopulationHistory:

    def __init__(self, n_individuals):
        self.individuals = {}
        for i in range(n_individuals):
            self.individuals[i] = IndividualHistory()

    def fill(self, individuals, fitness, times):
        for i in range(len(individuals)):
            self.individuals[i].set_values(individuals[i].sequence,
                                           float(fitness[i]), float(times[i]))

    def get_best_fitness(self):
        return (
            self.individuals[0].fitness["fitness"],
            self.individuals[0].fitness["time"],
        )

    def get_individuals(self):
        return self.individuals.items()


class PopulationWithElite(KartezioPopulation):

    def __init__(self):
        super().__init__()

    def set_elite(self, individual):
        self[0] = individual

    def get_elite(self):
        return self[0]

    def get_best_individual(self):
        # get the first element to minimize
        best_fitness_idx = np.argsort(self.score)[0]
        best_individual = self[best_fitness_idx]
        return best_individual, self.fitness[best_fitness_idx]

    def history(self):
        population_history = PopulationHistory(g._lambda + 1)
        population_history.fill(self.individuals, self.fitness, self.time)
        return population_history


class OnePlusLambda:

    def __init__(self, factory, init_method, mutation_method,
                 fitness):
        self._mu = 1
        self.factory = factory
        self.init_method = init_method
        self.mutation_method = mutation_method
        self.fitness = fitness
        self.population = PopulationWithElite()

    @property
    def elite(self):
        return self.population.get_elite()

    def initialization(self):
        for i in range(g._lambda + 1):
            individual = self.init_method.mutate(self.factory.create())
            self.population[i] = individual

    def selection(self):
        new_elite, fitness = self.population.get_best_individual()
        self.population.set_elite(new_elite)

    def reproduction(self):
        elite = self.population.get_elite()
        for i in range(self._mu, g._lambda + 1):
            self.population[i] = elite.clone()

    def mutation(self):
        for i in range(self._mu, g._lambda + 1):
            self.population[i] = self.mutation_method.mutate(
                self.population[i])

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.call(y_true, y_pred)
        self.population.set_fitness(fitness)

class G:
    pass


g = G()
g.path = "dataset"
g._lambda = 5
g.generations = 10
g.endpoint = EndpointWatershed()
g.bundle = BundleOpenCV()
g.inputs = 3
g.nodes = 30
g.outputs = 2
g.arity = 2
g.parameters = 2
node_mutation_rate = 0.15
output_mutation_rate = 0.2
g.out_idx = g.inputs + g.nodes
g.para_idx = 1 + g.arity
g.w = 1 + g.arity + g.parameters
g.h = g.inputs + g.nodes + g.outputs
g.prototype = KartezioGenome(shape=(g.h, g.w))
g.genome_factory = GenomeFactory(g.prototype)
g.parser = KartezioParser()
g.instance_method = MutationAllRandom(g.bundle.size)
mutation = MutationClassic(g.bundle.size,
                           node_mutation_rate,
                           output_mutation_rate)
g.mutation_method = GoldmanWrapper(mutation, g.parser)
g.fitness = FitnessAP()
g.strategy = OnePlusLambda(g.genome_factory, g.instance_method,
                         g.mutation_method, g.fitness)
model = ModelCGP(g.strategy, g.parser)
g.dataset_reader = DatasetReader(g.path, counting=False)
g.dataset = g.dataset_reader.read_dataset(dataset_filename=CSV_DATASET,
                                          meta_filename=JSON_META,
                                          indices=None)
g.callbacks = [
    CallbackVerbose(frequency=1),
    CallbackSave(".", g.dataset, frequency=1)
]
g.workdir = str(g.callbacks[1].workdir._path)
for callback in g.callbacks:
    callback.set_parser(model.parser)
    model.attach(callback)
model.fit(*g.dataset.train_xy)
pack_one_directory(g.workdir)
