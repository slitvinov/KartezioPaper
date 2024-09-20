from abc import ABC, abstractmethod
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
from numena.io.json import Serializable
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
from kartezio.model.registry import registry

Score = NewType("Score", float)
ScoreList = List[Score]


class KartezioComponent(Serializable, ABC):
    pass


class KartezioNode(KartezioComponent, ABC):

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

    @abstractmethod
    def call(self, x: List, args: List = None):
        pass

    def dumps(self) -> dict:
        return {
            "name": self.name,
            "abbv": self.symbol,
            "arity": self.arity,
            "args": self.args,
            "kwargs": self._to_json_kwargs(),
        }

    @abstractmethod
    def _to_json_kwargs(self) -> dict:
        pass


class KartezioEndpoint(KartezioNode, ABC):
    """
    Terminal KartezioNode, executed after graph parsing.
    Not submitted to evolution.
    """

    def __init__(self, name: str, symbol: str, arity: int, outputs_keys: list):
        super().__init__(name, symbol, arity, 0)
        self.outputs_keys = outputs_keys

    @staticmethod
    def from_json(json_data):
        return registry.endpoints.instantiate(json_data["abbv"],
                                              **json_data["kwargs"])


class KartezioPreprocessing(KartezioNode, ABC):
    """
    First KartezioNode, executed before evolution loop.
    Not submitted to evolution.
    """

    def __init__(self, name: str, symbol: str):
        super().__init__(name, symbol, 1, 0)


class KartezioBundle(KartezioComponent, ABC):

    def __init__(self):
        self.__nodes = {}
        self.fill()

    @staticmethod
    def from_json(json_data):
        bundle = EmptyBundle()
        for node_name in json_data:
            bundle.add_node(node_name)
        return bundle

    @abstractmethod
    def fill(self):
        pass

    def add_node(self, node_name):
        self.__nodes[len(self.__nodes)] = registry.nodes.instantiate(node_name)

    def add_bundle(self, bundle):
        for f in bundle.nodes:
            self.add_node(f.name)

    def name_of(self, i):
        return self.__nodes[i].name

    def symbol_of(self, i):
        return self.__nodes[i].symbol

    def arity_of(self, i):
        return self.__nodes[i].arity

    def parameters_of(self, i):
        return self.__nodes[i].p

    def execute(self, name, x, args):
        return self.__nodes[name].call(x, args)

    def show(self):
        for i, node in self.__nodes.items():
            print(f"[{i}] - {node.abbv}")

    @property
    def random_index(self):
        return random.choice(self.keys)

    @property
    def last_index(self):
        return len(self.__nodes) - 1

    @property
    def nodes(self):
        return list(self.__nodes.values())

    @property
    def keys(self):
        return list(self.__nodes.keys())

    @property
    def max_arity(self):
        return max([self.arity_of(i) for i in self.keys])

    @property
    def max_parameters(self):
        return max([self.parameters_of(i) for i in self.keys])

    @property
    def size(self):
        return len(self.__nodes)

    @property
    def ordered_list(self):
        return [self.__nodes[i].name for i in range(self.size)]

    def dumps(self) -> dict:
        return {}


class EmptyBundle(KartezioBundle):

    def fill(self):
        pass


class Prototype(ABC):
    """
    Using Prototype Pattern to duplicate:
    https://refactoring.guru/design-patterns/prototype
    """

    @abstractmethod
    def clone(self):
        pass


class KartezioGenome(KartezioComponent, Prototype):
    """
    Only store "DNA" in a numpy array
    No metadata stored in DNA to avoid duplicates
    Avoiding RAM overload: https://refactoring.guru/design-patterns/flyweight
    Default genome would be: 3 inputs, 10 function nodes (2 connections and 2 parameters), 1 output,
    so with shape (14, 5)
    Args:
        Prototype ([type]): [description]
    Returns:
        [type]: [description]
    """

    def dumps(self) -> dict:
        pass

    def __init__(self, shape: tuple = (14, 5), sequence: np.ndarray = None):
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = np.zeros(shape=shape, dtype=np.uint8)

    def __copy__(self):
        new = self.__class__(*self.sequence.shape)
        new.__dict__.update(self.__dict__)
        return new

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

    @staticmethod
    def from_json(json_data):
        sequence = np.asarray(ast.literal_eval(json_data["sequence"]))
        return KartezioGenome(sequence=sequence)


def to_metadata(json_data):
    return GenomeShape(
        json_data["n_in"],
        json_data["columns"],
        json_data["n_out"],
        json_data["n_conn"],
        json_data["n_para"],
    )


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


class GenomeFactory(Factory):

    def __init__(self, prototype: KartezioGenome):
        super().__init__(prototype)


class GenomeAdapter(KartezioComponent, ABC):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    def __init__(self, shape):
        self.shape = shape


class GenomeWriter(GenomeAdapter):

    def write_function(self, genome, node, function_id):
        genome[self.shape.nodes_idx + node, self.shape.func_idx] = function_id

    def write_connections(self, genome, node, connections):
        genome[self.shape.nodes_idx + node,
               self.shape.con_idx:self.shape.para_idx] = connections

    def write_parameters(self, genome, node, parameters):
        genome[self.shape.nodes_idx + node, self.shape.para_idx:] = parameters

    def write_output_connection(self, genome, output_index, connection):
        genome[self.shape.out_idx + output_index,
               self.shape.con_idx] = connection


class GenomeReader(GenomeAdapter):

    def read_function(self, genome, node):
        return genome[self.shape.nodes_idx + node, self.shape.func_idx]

    def read_connections(self, genome, node):
        return genome[self.shape.nodes_idx + node,
                      self.shape.con_idx:self.shape.para_idx]

    def read_active_connections(self, genome, node, active_connections):
        return genome[
            self.shape.nodes_idx + node,
            self.shape.con_idx:self.shape.con_idx + active_connections,
        ]

    def read_parameters(self, genome, node):
        return genome[self.shape.nodes_idx + node, self.shape.para_idx:]

    def read_outputs(self, genome):
        return genome[self.shape.out_idx:, :]


class GenomeReaderWriter(GenomeReader, GenomeWriter):
    pass


@dataclass
class GenomeShape:
    inputs: int = 3
    nodes: int = 10
    outputs: int = 1
    connections: int = 2
    parameters: int = 2
    in_idx: int = field(init=False, repr=False)
    func_idx: int = field(init=False, repr=False)
    con_idx: int = field(init=False, repr=False)
    nodes_idx = None
    out_idx = None
    para_idx = None
    w: int = field(init=False)
    h: int = field(init=False)
    prototype = None

    def __post_init__(self):
        self.in_idx = 0
        self.func_idx = 0
        self.con_idx = 1
        self.nodes_idx = self.inputs
        self.out_idx = self.nodes_idx + self.nodes
        self.para_idx = self.con_idx + self.connections
        self.w = 1 + self.connections + self.parameters
        self.h = self.inputs + self.nodes + self.outputs
        self.prototype = KartezioGenome(shape=(self.h, self.w))

    @staticmethod
    def from_json(json_data):
        return GenomeShape(
            json_data["n_in"],
            json_data["columns"],
            json_data["n_out"],
            json_data["n_conn"],
            json_data["n_para"],
        )


class KartezioParser(GenomeReader):

    def __init__(self, shape, function_bundle, endpoint):
        super().__init__(shape)
        self.function_bundle = function_bundle
        self.endpoint = endpoint

    def to_series_parser(self, stacker):
        return ParserChain(self.shape, self.function_bundle, stacker,
                           self.endpoint)

    def dumps(self) -> dict:
        return {
            "metadata": {
                "rows": 1,  # single row CGP
                "columns": self.shape.nodes,
                "n_in": self.shape.inputs,
                "n_out": self.shape.outputs,
                "n_para": self.shape.parameters,
                "n_conn": self.shape.connections,
            },
            "functions": self.function_bundle.ordered_list,
            "endpoint": self.endpoint.dumps(),
            "mode": "default",
        }

    @staticmethod
    def from_json(json_data):
        shape = GenomeShape.from_json(json_data["metadata"])
        bundle = KartezioBundle.from_json(json_data["functions"])
        endpoint = KartezioEndpoint.from_json(json_data["endpoint"])
        if json_data["mode"] == "series":
            stacker = KartezioStacker.from_json(json_data["stacker"])
            return ParserChain(shape, bundle, stacker, endpoint)
        return KartezioParser(shape, bundle, endpoint)

    def _parse_one_graph(self, genome, graph_source):
        next_indices = graph_source.copy()
        output_tree = graph_source.copy()
        while next_indices:
            next_index = next_indices.pop()
            if next_index < self.shape.inputs:
                continue
            function_index = self.read_function(genome,
                                                next_index - self.shape.inputs)
            active_connections = self.function_bundle.arity_of(function_index)
            next_connections = set(
                self.read_active_connections(genome,
                                             next_index - self.shape.inputs,
                                             active_connections))
            next_indices = next_indices.union(next_connections)
            output_tree = output_tree.union(next_connections)
        return sorted(list(output_tree))

    def parse_to_graphs(self, genome):
        outputs = self.read_outputs(genome)
        graphs_list = [
            self._parse_one_graph(genome, {output[self.shape.con_idx]})
            for output in outputs
        ]
        return graphs_list

    def _x_to_output_map(self, genome: KartezioGenome, graphs_list: List,
                         x: List):
        output_map = {i: x[i].copy() for i in range(self.shape.inputs)}
        for graph in graphs_list:
            for node in graph:
                # inputs are already in the map
                if node < self.shape.inputs:
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                arity = self.function_bundle.arity_of(function_index)
                connections = self.read_active_connections(
                    genome, node_index, arity)
                inputs = [output_map[c] for c in connections]
                p = self.read_parameters(genome, node_index)
                value = self.function_bundle.execute(function_index, inputs, p)
                output_map[node] = value
        return output_map

    def _parse_one(self, genome: KartezioGenome, graphs_list: List, x: List):
        # fill output_map with inputs
        output_map = self._x_to_output_map(genome, graphs_list, x)
        return [
            output_map[output_gene[self.shape.con_idx]]
            for output_gene in self.read_outputs(genome)
        ]

    def active_size(self, genome):
        node_list = []
        graphs_list = self.parse_to_graphs(genome)
        for graph in graphs_list:
            for node in graph:
                if node < self.shape.inputs:
                    continue
                if node < self.shape.out_idx:
                    node_list.append(node)
                else:
                    continue
        return len(node_list)

    def node_histogram(self, genome):
        nodes = {}
        graphs_list = self.parse_to_graphs(genome)
        for graph in graphs_list:
            for node in graph:
                # inputs are already in the map
                if node < self.shape.inputs:
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.function_bundle.symbol_of(function_index)
                if function_name not in nodes.keys():
                    nodes[function_name] = 0
                nodes[function_name] += 1
        return nodes

    def get_last_node(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        output_functions = []
        for graph in graphs_list:
            for node in graph[-1:]:
                # inputs are already in the map
                if node < self.shape.inputs:
                    print(f"output {node} directly connected to input.")
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.function_bundle.symbol_of(function_index)
                output_functions.append(function_name)
        return output_functions

    def get_first_node(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        input_functions = []
        for graph in graphs_list:
            for node in graph:
                if node < self.shape.inputs:
                    print(f"output {node} directly connected to input.")
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.function_bundle.symbol_of(function_index)
                arity = self.function_bundle.arity_of(function_index)
                connections = self.read_active_connections(
                    genome, node_index, arity)
                for c in connections:
                    if c < self.shape.inputs:
                        input_functions.append(function_name)
        return input_functions

    def bigrams(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        outputs = self.read_outputs(genome)
        print(graphs_list)
        bigram_list = []
        for i, graph in enumerate(graphs_list):
            for j, node in enumerate(graph):
                if node < self.shape.inputs:
                    continue
                node_index = node - self.shape.inputs
                function_index = self.read_function(genome, node_index)
                fname = self.function_bundle.symbol_of(function_index)
                arity = self.function_bundle.arity_of(function_index)
                connections = self.read_active_connections(
                    genome, node_index, arity)
                for k, c in enumerate(connections):
                    if c < self.shape.inputs:
                        in_name = f"IN-{c}"
                        pair = (f"{fname}", in_name)
                        """
                        if arity == 1:
                            pair = (f"{fname}", in_name)
                        else:
                            pair = (f"{fname}-{k}", in_name)
                        """
                    else:
                        f2_index = self.read_function(genome,
                                                      c - self.shape.inputs)
                        f2_name = self.function_bundle.symbol_of(f2_index)
                        """
                        if arity == 1:
                            pair = (f"{fname}", f2_name)
                        else:
                            pair = (f"{fname}-{k}", f2_name)
                        """
                        pair = (f"{fname}", f2_name)
                    bigram_list.append(pair)
            f_last = self.read_function(genome,
                                        outputs[i][1] - self.shape.inputs)
            fname = self.function_bundle.symbol_of(f_last)
            pair = (f"OUT-{i}", fname)
            bigram_list.append(pair)
        print(bigram_list)
        return bigram_list

    def function_distribution(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        active_list = []
        for graph in graphs_list:
            for node in graph:
                if node < self.shape.inputs:
                    continue
                if node >= self.shape.out_idx:
                    continue
                active_list.append(node)
        functions = []
        is_active = []
        for i, _ in enumerate(genome.sequence):
            if i < self.shape.inputs:
                continue
            if i >= self.shape.out_idx:
                continue
            node_index = i - self.shape.inputs
            function_index = self.read_function(genome, node_index)
            function_name = self.function_bundle.symbol_of(function_index)
            functions.append(function_name)
            is_active.append(i in active_list)
        return functions, is_active

    def parse_population(self, population, x):
        y_pred = []
        for i in range(len(population.individuals)):
            y, t = self.parse(population.individuals[i], x)
            population.set_time(i, t)
            y_pred.append(y)
        return y_pred

    def parse(self, genome, x):
        """Decode the Genome given a list of inputs
        Args:
            genome (KartezioGenome): [description]
            x (List): [description]
        Returns:
            [type]: [description]
        """
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genome)
        # for each image
        for xi in x:
            start_time = time.time()
            y_pred = self._parse_one(genome, graphs, xi)
            if self.endpoint is not None:
                y_pred = self.endpoint.call(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time


class ParserSequential(KartezioParser):
    """TODO: default Parser, KartezioParser becomes ABC"""
    pass


class ParserChain(KartezioParser):

    def __init__(self, shape, bundle, stacker, endpoint):
        super().__init__(shape, bundle, endpoint)
        self.stacker = stacker

    def parse(self, genome, x):
        """Decode the Genome given a list of inputs
        Args:
            genome (KartezioGenome): [description]
            x (List): [description]
        Returns:
            [type]: [description]
        """
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genome)
        for series in x:
            start_time = time.time()
            y_pred_series = []
            # for each image
            for xi in series:
                y_pred = self._parse_one(genome, graphs, xi)
                y_pred_series.append(y_pred)
            y_pred = self.endpoint.call(self.stacker.call(y_pred_series))
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time

    def dumps(self) -> dict:
        json_data = super().dumps()
        json_data["mode"] = "series"
        json_data["stacker"] = self.stacker.dumps()
        return json_data


class KartezioToCode(KartezioParser):

    def to_python_class(self, node_name, genome):
        pass


class KartezioStacker(KartezioNode, ABC):

    def __init__(self, name: str, symbol: str, arity: int):
        super().__init__(name, symbol, arity, 0)

    def call(self, x: List, args: List = None):
        y = []
        for i in range(self.arity):
            Y = [xi[i] for xi in x]
            y.append(self.post_stack(self.stack(Y), i))
        return y

    @abstractmethod
    def stack(self, Y: List):
        pass

    def post_stack(self, x, output_index):
        return x

    @staticmethod
    def from_json(json_data):
        return registry.stackers.instantiate(json_data["abbv"],
                                             arity=json_data["arity"],
                                             **json_data["kwargs"])


class ExportableNode(KartezioNode, ABC):

    def _to_json_kwargs(self) -> dict:
        return {}

    @abstractmethod
    def to_python(self, input_nodes: List, p: List, node_name: str):
        """
        Parameters
        ----------
        input_nodes :
        p :
        node_name :
        """
        pass

    @abstractmethod
    def to_cpp(self, input_nodes: List, p: List, node_name: str):
        """
        :param input_nodes:
        :type input_nodes:
        :param p:
        :type p:
        :param node_name:
        :type node_name:
        """
        pass


class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """

    @abstractmethod
    def update(self, event):
        """
        Receive update from subject.
        """
        pass


class KartezioCallback(KartezioComponent, Observer, ABC):

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.parser = None

    def set_parser(self, parser):
        self.parser = parser

    def update(self, event):
        if event["n"] % self.frequency == 0 or event["force"]:
            self._callback(event["n"], event["name"], event["content"])

    def dumps(self) -> dict:
        return {}

    @abstractmethod
    def _callback(self, n, e_name, e_content):
        pass


class NodeImageProcessing(KartezioNode, ABC):

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

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.add({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::add({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("subtract")
class Subtract(ExportableNode):

    def __init__(self):
        super().__init__("subtract", "SUB", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.subtract(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.subtract({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::subtract({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("bitwise_not")
class BitwiseNot(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_not", "NOT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_not(x[0])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_not({input_names[0]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_not({input_names[0]}, {output_name});"


@registry.nodes.add("bitwise_or")
class BitwiseOr(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_or", "BOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_or(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_or({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_or({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("bitwise_and")
class BitwiseAnd(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_and", "BAND", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_and({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_and({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("bitwise_and_mask")
class BitwiseAndMask(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_and_mask", "ANDM", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[0], mask=x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_and({input_names[0]}, {input_names[0]}, mask={input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_and({input_names[0]}, {input_names[0]}, {output_name}, {input_names[1]});"


@registry.nodes.add("bitwise_xor")
class BitwiseXor(ExportableNode):

    def __init__(self):
        super().__init__("bitwise_xor", "BXOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_xor(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_xor({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_xor({input_names[0]}, {input_names[1]}, {output_name});"


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

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.erode({input_names[0]}, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::erode({input_names[0]}, {output_name}, kernel_from_parameters({args[0]}));"


@registry.nodes.add("dilate")
class Dilate(ExportableNode):

    def __init__(self):
        super().__init__("dilate", "DILT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.dilate(inputs[0], kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.dilate({input_names[0]}, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::dilate({input_names[0]}, {output_name}, kernel_from_parameters({args[0]}));"


@registry.nodes.add("open")
class Open(ExportableNode):

    def __init__(self):
        super().__init__("open", "OPEN", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_OPEN, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_OPEN, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_OPEN, kernel_from_parameters({args[0]}));"


@registry.nodes.add("close")
class Close(ExportableNode):

    def __init__(self):
        super().__init__("close", "CLSE", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_CLOSE, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_CLOSE, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_CLOSE, kernel_from_parameters({args[0]}));"


@registry.nodes.add("morph_gradient")
class MorphGradient(ExportableNode):

    def __init__(self):
        super().__init__("morph_gradient", "MGRD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_GRADIENT, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_GRADIENT, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_GRADIENT, kernel_from_parameters({args[0]}));"


@registry.nodes.add("morph_tophat")
class MorphTopHat(ExportableNode):

    def __init__(self):
        super().__init__("morph_tophat", "MTHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_TOPHAT, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_TOPHAT, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_TOPHAT, kernel_from_parameters({args[0]}));"


@registry.nodes.add("morph_blackhat")
class MorphBlackHat(ExportableNode):

    def __init__(self):
        super().__init__("morph_blackhat", "MBHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_BLACKHAT, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_BLACKHAT, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_BLACKHAT, kernel_from_parameters({args[0]}));"


@registry.nodes.add("fill_holes")
class FillHoles(ExportableNode):

    def __init__(self):
        super().__init__("fill_holes", "FILL", 1, 0, sources="Handmade")

    def call(self, inputs, p):
        return morph_fill(inputs[0])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = imfill({input_names[0]})"

    def to_cpp(self, input_names, p, output_name):
        return "imfill({input_names[0]}, {output_name});"


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


def to_genome(json_data):
    sequence = np.asarray(ast.literal_eval(json_data["sequence"]))
    return KartezioGenome(sequence=sequence)


def from_individual(individual):
    return {
        "sequence": simplejson.dumps(individual.sequence.tolist()),
        "fitness": individual.fitness,
    }


def from_population(population: List):
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


def singleton(cls):
    """
    https://towardsdatascience.com/10-fabulous-python-decorators-ab674a732871
    """
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


class Observable(ABC):
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def notify(self, event) -> None:
        for observer in self._observers:
            observer.update(event)


def register_endpoints():
    print(
        f"[Kartezio - INFO] -  {len(registry.endpoints.list())} endpoints registered."
    )


class TransformToHSV(KartezioPreprocessing):

    def __init__(self, source_color="bgr"):
        super().__init__("Transform to HSV", "HSV")
        self.source_color = source_color

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            if self.source_color == "bgr":
                transformed = bgr2hsv(original_image)
            elif self.source_color == "rgb":
                transformed = bgr2hsv(rgb2bgr(original_image))
            new_x.append(image_split(transformed))
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass


class TransformToHED(KartezioPreprocessing):

    def __init__(self, source_color="bgr"):
        super().__init__("Transform to HED", "HED")
        self.source_color = source_color

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            if self.source_color == "bgr":
                transformed = bgr2hed(original_image)
            elif self.source_color == "rgb":
                transformed = rgb2hed(original_image)
            new_x.append(image_split(transformed))
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass


class SelectChannels(KartezioPreprocessing):

    def __init__(self, channels):
        super().__init__("Channel Selection", "CHAN")
        self.channels = channels

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            one_item = [x[i][channel] for channel in self.channels]
            new_x.append(one_item)
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass


class Format3D(KartezioPreprocessing):

    def __init__(self, channels=None, z_range=None):
        super().__init__("Format to 3D", "F3D")
        self.channels = channels
        self.z_range = z_range

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            one_item = []
            if self.channels:
                if self.z_range:
                    for z in self.z_range:
                        one_item.append(
                            [x[i][channel][z] for channel in self.channels])
                else:
                    for z in range(len(x[i][0])):
                        one_item.append(
                            [x[i][channel][z] for channel in self.channels])
            else:
                if self.z_range:
                    for z in self.z_range:
                        one_item.append([x[i][:][z]])
                else:
                    for z in range(len(x[i])):
                        one_item.append([x[i][:][z]])
            new_x.append(one_item)
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass


@singleton
class BundleOpenCV(KartezioBundle):

    def fill(self):
        for node_abbv in IMAGE_NODES_ABBV_LIST:
            self.add_node(node_abbv)


BUNDLE_OPENCV = BundleOpenCV()


def register_stackers():
    print(
        f"[Kartezio - INFO] -  {len(registry.stackers.list())} stackers registered."
    )


@registry.stackers.add("MEAN")
class StackerMean(KartezioStacker):

    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self,
                 name="mean_stacker",
                 symbol="MEAN",
                 arity=1,
                 threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y: List):
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

    def stack(self, Y: List):
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

    def stack(self, Y: List):
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

    def stack(self, Y: List):
        return np.max(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 0:
            return cv2.GaussianBlur(yi, (7, 7), 1)
        return yi


@registry.stackers.add("MEANW")
class MeanKartezioStackerForWatershed(KartezioStacker):

    def _to_json_kwargs(self) -> dict:
        return {
            "half_kernel_size": self.half_kernel_size,
            "threshold": self.threshold
        }

    def __init__(self, half_kernel_size=1, threshold=4):
        super().__init__(name="mean_stacker_watershed",
                         symbol="MEANW",
                         arity=2)
        self.half_kernel_size = half_kernel_size
        self.threshold = threshold

    def stack(self, Y: List):
        return np.mean(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 1:
            # supposed markers
            yi = morph_erode(yi, half_kernel_size=self.half_kernel_size)
        return threshold_tozero(yi, self.threshold)


JSON_ELITE = "elite.json"
JSON_HISTORY = "history.json"
JSON_META = "META.json"
CSV_DATASET = "dataset.csv"
DIR_PREVIEW = "__preview__"


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

    @abstractmethod
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


@registry.readers.add("image_mask")
class ImageMaskReader(DataReader):

    def _read(self, filepath, shape=None):
        if filepath == "":
            mask = image_new(shape)
            return DataItem([mask], shape, 0)
        image = imread_grayscale(filepath)
        _, labels = cv2.connectedComponents(image)
        return DataItem([labels], image.shape[:2],
                        len(np.unique(labels)) - 1, image)


@registry.readers.add("image_hsv")
class ImageHSVReader(DataReader):

    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_hsv = bgr2hsv(image_bgr)
        return DataItem(image_split(image_hsv), image_bgr.shape[:2], None,
                        image_bgr)


@registry.readers.add("image_hed")
class ImageHEDReader(DataReader):

    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_hed = bgr2hed(image_bgr)
        return DataItem(image_split(image_hed), image_bgr.shape[:2], None,
                        image_bgr)


@registry.readers.add("image_labels")
class ImageLabels(DataReader):

    def _read(self, filepath, shape=None):
        image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        for i, current_value in enumerate(np.unique(image)):
            image[image == current_value] = i
        return DataItem([image], image.shape[:2], image.max(), visual=image)


@registry.readers.add("image_rgb")
class ImageRGBReader(DataReader):

    def _read(self, filepath, shape=None):
        image = imread_color(filepath, rgb=False)
        return DataItem(image_split(image),
                        image.shape[:2],
                        None,
                        visual=rgb2bgr(image))


@registry.readers.add("csv_ellipse")
class CsvEllipseReader(DataReader):

    def _read(self, filepath, shape=None):
        dataframe = pd.read_csv(filepath)
        ellipses = read_ellipses_from_csv(dataframe,
                                          scale=self.scale,
                                          ellipse_scale=1.0)
        label_mask = image_new(shape)
        fill_ellipses_as_labels(label_mask, ellipses)
        return DataItem([label_mask], shape, len(ellipses))


@registry.readers.add("image_grayscale")
class ImageGrayscaleReader(DataReader):

    def _read(self, filepath, shape=None):
        image = imread_grayscale(filepath)
        visual = cv2.merge((image, image, image))
        return DataItem([image], image.shape, None, visual=visual)


@registry.readers.add("roi_polygon")
class RoiPolygonReader(DataReader):

    def _read(self, filepath, shape=None):
        label_mask = image_new(shape)
        if filepath == "":
            return DataItem([label_mask], shape, 0)
        polygons = read_polygons_from_roi(filepath)
        fill_polygons_as_labels(label_mask, polygons)
        return DataItem([label_mask], shape, len(polygons))


@registry.readers.add("one-hot_vector")
class OneHotVectorReader(DataReader):

    def _read(self, filepath, shape=None):
        label = np.array(ast.literal_eval(filepath.split("/")[-1]))
        return DataItem([label], shape, None)


@registry.readers.add("image_channels")
class ImageChannelsReader(DataReader):

    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = image.shape[-2:]
        if len(image.shape) == 2:
            channels = [image]
            preview = gray2rgb(channels[0])
        if len(image.shape) == 3:
            # channels: (c, h, w)
            channels = [channel for channel in image]
            preview = cv2.merge(
                (image_new(channels[0].shape), channels[0], channels[1]))
        if len(image.shape) == 4:
            # stack: (z, c, h, w)
            channels = [image[:, i] for i in range(len(image[0]))]
            preview = cv2.merge((
                channels[0].max(axis=0).astype(np.uint8),
                channels[1].max(axis=0).astype(np.uint8),
                image_new(channels[0][0].shape, dtype=np.uint8),
            ))
            cv2.imwrite("rgb_image.png", preview)
        return DataItem(channels, shape, None, visual=preview)


@dataclass
class DatasetReader(Directory):
    counting: bool = False
    preview: bool = False
    preview_dir: Directory = field(init=False)

    def __post_init__(self, path):
        super().__post_init__(path)
        if self.preview:
            self.preview_dir = self.next(DIR_PREVIEW)

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
        if not np.all((input_sizes == inputs)):
            """
            raise ValueError(
                f"Inconsistent size of inputs for this dataset: sizes: {input_sizes}"
            )
            """
            print(
                f"Inconsistent size of inputs for this dataset: sizes: {input_sizes}"
            )
        if self.preview:
            for i in range(len(training.x)):
                visual = training.v[i]
                label = training.y[i][0]
                preview = draw_overlay(visual,
                                       label.astype(np.uint8),
                                       color=[224, 255, 255],
                                       alpha=0.5)
                self.preview_dir.write(f"train_{i}.png", preview)
            for i in range(len(testing.x)):
                visual = testing.v[i]
                label = testing.y[i][0]
                preview = draw_overlay(visual,
                                       label.astype(np.uint8),
                                       color=[224, 255, 255],
                                       alpha=0.5)
                self.preview_dir.write(f"test_{i}.png", preview)
        return Dataset(training, testing, self.name, self.label_name, inputs,
                       indices)

    def _read_auto(self, dataset):
        pass

    def _read_dataset(self, dataframe, indices=None):
        dataset = Dataset.SubSet(dataframe)
        dataframe.reset_index(inplace=True)
        if indices:
            dataframe = dataframe.loc[indices]
        for row in dataframe.itertuples():
            x = self.input_reader.read(row.input, shape=None)
            y = self.label_reader.read(row.label, shape=x.shape)
            if self.counting:
                y = [y.datalist[0], y.count]
            else:
                y = y.datalist
            dataset.n_inputs = x.size
            dataset.add_item(x.datalist, y)
            visual_from_table = False
            if "visual" in dataframe.columns:
                if str(row.visual) != "nan":
                    dataset.add_visual(self.read(row.visual))
                    visual_from_table = True
            if not visual_from_table:
                dataset.add_visual(x.visual)
        return dataset


def read_dataset(
    dataset_path,
    filename=CSV_DATASET,
    meta_filename=JSON_META,
    indices=None,
    counting=False,
    preview=False,
    reader=None,
):
    dataset_reader = DatasetReader(dataset_path,
                                   counting=counting,
                                   preview=preview)
    if reader is not None:
        dataset_reader.add_reader(reader)
    return dataset_reader.read_dataset(dataset_filename=filename,
                                       meta_filename=meta_filename,
                                       indices=indices)


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


class KartezioMetric(KartezioNode, ABC):

    def __init__(
        self,
        name: str,
        symbol: str,
        arity: int,
    ):
        super().__init__(name, symbol, arity, 0)

    def _to_json_kwargs(self) -> dict:
        pass


class KartezioFitness(KartezioNode, ABC):

    def __init__(
        self,
        name: str,
        symbol: str,
        arity: int,
        default_metric: KartezioMetric = None,
    ):
        super().__init__(name, symbol, arity, 0)
        self.metrics: MetricList = []
        if default_metric:
            self.add_metric(default_metric)

    def add_metric(self, metric: KartezioMetric):
        self.metrics.append(metric)

    def call(self, y_true, y_pred) -> ScoreList:
        scores: ScoreList = []
        for yi_pred in y_pred:
            scores.append(self.compute_one(y_true, yi_pred))
        return scores

    def compute_one(self, y_true, y_pred) -> Score:
        score = 0.0
        y_size = len(y_true)
        for i in range(y_size):
            _y_true = y_true[i].copy()
            _y_pred = y_pred[i]
            score += self.__fitness_sum(_y_true, _y_pred)
        return Score(score / y_size)

    def __fitness_sum(self, y_true, y_pred) -> Score:
        score = Score(0.0)
        for metric in self.metrics:
            score += metric.call(y_true, y_pred)
        return score

    def _to_json_kwargs(self) -> dict:
        pass


class KartezioMutation(GenomeReaderWriter, ABC):

    def __init__(self, shape, n_functions):
        super().__init__(shape)
        self.n_functions = n_functions
        self.parameter_max_value = 256

    def dumps(self) -> dict:
        return {}

    @property
    def random_parameters(self):
        return np.random.randint(self.parameter_max_value,
                                 size=self.shape.parameters)

    @property
    def random_functions(self):
        return np.random.randint(self.n_functions)

    @property
    def random_output(self):
        return np.random.randint(self.shape.out_idx, size=1)

    def random_connections(self, idx: int):
        return np.random.randint(self.shape.nodes_idx + idx,
                                 size=self.shape.connections)

    def mutate_function(self, genome: KartezioGenome, idx: int):
        self.write_function(genome, idx, self.random_functions)

    def mutate_connections(self,
                           genome: KartezioGenome,
                           idx: int,
                           only_one: int = None):
        new_connections = self.random_connections(idx)
        if only_one is not None:
            new_value = new_connections[only_one]
            new_connections = self.read_connections(genome, idx)
            new_connections[only_one] = new_value
        self.write_connections(genome, idx, new_connections)

    def mutate_parameters(self,
                          genome: KartezioGenome,
                          idx: int,
                          only_one: int = None):
        new_parameters = self.random_parameters
        if only_one is not None:
            old_parameters = self.read_parameters(genome, idx)
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.write_parameters(genome, idx, new_parameters)

    def mutate_output(self, genome: KartezioGenome, idx: int):
        self.write_output_connection(genome, idx, self.random_output)

    @abstractmethod
    def mutate(self, genome: KartezioGenome):
        pass


class KartezioES(ABC):

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def reproduction(self):
        pass


@registry.endpoints.add("TRSH")
class EndpointThreshold(KartezioEndpoint):

    def __init__(self, threshold=1):
        super().__init__(f"Threshold (t={threshold})", "TRSH", 1, ["mask"])
        self.threshold = threshold

    def call(self, x, args=None):
        mask = x[0].copy()
        mask[mask < self.threshold] = 0
        return {"mask": mask}

    def _to_json_kwargs(self) -> dict:
        return {"threshold": self.threshold}


@registry.endpoints.add("WSHD")
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


class JsonLoader:

    def read_individual(self, filepath):
        json_data = json_read(filepath=filepath)
        dataset = json_data["dataset"]
        parser = KartezioParser.from_json(json_data["decoding"])
        try:
            individual = KartezioGenome.from_json(json_data["individual"])
        except KeyError:
            try:
                individual = KartezioGenome.from_json(json_data)
            except KeyError:
                individual = KartezioGenome.from_json(
                    json_data["population"][0])
        return dataset, individual, parser


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


class IndividualHistory:

    def __init__(self):
        self.fitness = {"fitness": 0.0, "time": 0.0}
        self.sequence = None

    def set_sequence(self, sequence):
        self.sequence = sequence

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


class KartezioPopulation(KartezioComponent, ABC):

    def __init__(self, size):
        self.size = size
        self.individuals = [None] * self.size
        self._fitness = {
            "fitness": np.zeros(self.size),
            "time": np.zeros(self.size)
        }

    def dumps(self) -> dict:
        return {}

    @abstractmethod
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

    def has_best_fitness(self):
        return min(self.fitness) == 0.0

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


class PopulationWithElite(KartezioPopulation):

    def __init__(self, _lambda):
        super().__init__(1 + _lambda)

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
        population_history = PopulationHistory(self.size)
        population_history.fill(self.individuals, self.fitness, self.time)
        return population_history


class OnePlusLambda(KartezioES):

    def __init__(self, _lambda, factory, init_method, mutation_method,
                 fitness):
        self._mu = 1
        self._lambda = _lambda
        self.factory = factory
        self.init_method = init_method
        self.mutation_method = mutation_method
        self.fitness = fitness
        self.population = PopulationWithElite(_lambda)

    @property
    def elite(self):
        return self.population.get_elite()

    def initialization(self):
        for i in range(self.population.size):
            individual = self.init_method.mutate(self.factory.create())
            self.population[i] = individual

    def selection(self):
        new_elite, fitness = self.population.get_best_individual()
        self.population.set_elite(new_elite)

    def reproduction(self):
        elite = self.population.get_elite()
        for i in range(self._mu, self.population.size):
            self.population[i] = elite.clone()

    def mutation(self):
        for i in range(self._mu, self.population.size):
            self.population[i] = self.mutation_method.mutate(
                self.population[i])

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.call(y_true, y_pred)
        self.population.set_fitness(fitness)


@dataclass
class ModelContext:
    genome_shape: GenomeShape = field(init=False)
    genome_factory: GenomeFactory = field(init=False)
    instance_method: KartezioMutation = field(init=False)
    mutation_method: KartezioMutation = field(init=False)
    fitness: KartezioFitness = field(init=False)
    stacker: KartezioStacker = field(init=False)
    endpoint: KartezioEndpoint = field(init=False)
    bundle: KartezioBundle = field(init=False)
    parser: KartezioParser = field(init=False)
    series_mode: bool = field(init=False)
    inputs: InitVar[int] = 3
    nodes: InitVar[int] = 10
    outputs: InitVar[int] = 1
    arity: InitVar[int] = 2
    parameters: InitVar[int] = 2

    def __post_init__(self, inputs: int, nodes: int, outputs: int, arity: int,
                      parameters: int):
        self.genome_shape = GenomeShape(inputs, nodes, outputs, arity,
                                        parameters)
        self.genome_factory = GenomeFactory(self.genome_shape.prototype)

    def set_bundle(self, bundle: KartezioBundle):
        self.bundle = bundle

    def set_endpoint(self, endpoint: KartezioEndpoint):
        self.endpoint = endpoint

    def set_instance_method(self, instance_method):
        self.instance_method = instance_method

    def set_mutation_method(self, mutation_method):
        self.mutation_method = mutation_method

    def set_fitness(self, fitness):
        self.fitness = fitness

    def compile_parser(self, series_mode, series_stacker):
        if series_mode:
            if type(series_stacker) == str:
                series_stacker = registry.stackers.instantiate(series_stacker)
            parser = ParserChain(self.genome_shape, self.bundle,
                                 series_stacker, self.endpoint)
        else:
            parser = KartezioParser(self.genome_shape, self.bundle,
                                    self.endpoint)
        self.parser = parser
        self.series_mode = series_mode


class GoldmanWrapper(KartezioMutation):

    def __init__(self, mutation, decoder):
        super().__init__(None, None)
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


@registry.mutations.add("classic")
class MutationClassic(KartezioMutation):

    def __init__(self, shape, n_functions, mutation_rate,
                 output_mutation_rate):
        super().__init__(shape, n_functions)
        self.mutation_rate = mutation_rate
        self.output_mutation_rate = output_mutation_rate
        self.n_mutations = int(
            np.floor(self.shape.nodes * self.shape.w * self.mutation_rate))
        self.all_indices = np.indices((self.shape.nodes, self.shape.w))
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
            elif mutation_parameter_index <= self.shape.connections:
                connection_idx = mutation_parameter_index - 1
                self.mutate_connections(genome, idx, only_one=connection_idx)
            else:
                parameter_idx = mutation_parameter_index - self.shape.connections - 1
                self.mutate_parameters(genome, idx, only_one=parameter_idx)
        for output in range(self.shape.outputs):
            if random.random() < self.output_mutation_rate:
                self.mutate_output(genome, output)
        return genome


@registry.mutations.add("all_random")
class MutationAllRandom(KartezioMutation):
    """
    Can be used to initialize genome (genome) randomly
    """

    def __init__(self, metadata: GenomeShape, n_functions: int):
        super().__init__(metadata, n_functions)

    def mutate(self, genome: KartezioGenome):
        # mutate genes
        for i in range(self.shape.nodes):
            self.mutate_function(genome, i)
            self.mutate_connections(genome, i)
            self.mutate_parameters(genome, i)
        # mutate outputs
        for i in range(self.shape.outputs):
            self.mutate_output(genome, i)
        return genome


@registry.mutations.add("copy")
class CopyGenome:

    def __init__(self, genome: KartezioGenome):
        self.genome = genome

    def mutate(self, _genome: KartezioGenome):
        return self.genome.clone()


class ModelBuilder:

    def __init__(self):
        self.__context = None

    def create(
            self,
            endpoint,
            bundle,
            inputs=3,
            nodes=10,
            outputs=1,
            arity=2,
            parameters=2,
            series_mode=False,
            series_stacker=StackerMean(),
    ):
        self.__context = ModelContext(inputs, nodes, outputs, arity,
                                      parameters)
        self.__context.set_endpoint(endpoint)
        self.__context.set_bundle(bundle)
        self.__context.compile_parser(series_mode, series_stacker)

    def set_instance_method(self, instance_method):
        if type(instance_method) == str:
            if instance_method == "random":
                shape = self.__context.genome_shape
                n_nodes = self.__context.bundle.size
                instance_method = MutationAllRandom(shape, n_nodes)
        self.__context.set_instance_method(instance_method)

    def set_mutation_method(self,
                            mutation,
                            node_mutation_rate,
                            output_mutation_rate,
                            use_goldman=True):
        if type(mutation) == str:
            shape = self.__context.genome_shape
            n_nodes = self.__context.bundle.size
            mutation = registry.mutations.instantiate(mutation, shape, n_nodes,
                                                      node_mutation_rate,
                                                      output_mutation_rate)
        if use_goldman:
            parser = self.__context.parser
            mutation = GoldmanWrapper(mutation, parser)
        self.__context.set_mutation_method(mutation)

    def set_fitness(self, fitness):
        if type(fitness) == str:
            fitness = registry.fitness.instantiate(fitness)
        self.__context.set_fitness(fitness)

    def compile(self,
                generations,
                _lambda,
                callbacks=None,
                dataset_inputs=None):
        factory = self.__context.genome_factory
        instance_method = self.__context.instance_method
        mutation_method = self.__context.mutation_method
        fitness = self.__context.fitness
        parser = self.__context.parser
        if parser.endpoint.arity != parser.shape.outputs:
            raise ValueError(
                f"Endpoint [{parser.endpoint.name}] requires {parser.endpoint.arity} output nodes. ({parser.shape.outputs} given)"
            )
        if self.__context.series_mode:
            if not isinstance(parser.stacker, KartezioStacker):
                raise ValueError(
                    f"Stacker {parser.stacker} has not been properly set.")
            if parser.stacker.arity != parser.shape.outputs:
                raise ValueError(
                    f"Stacker [{parser.stacker.name}] requires {parser.stacker.arity} output nodes. ({parser.shape.outputs} given)"
                )
        if dataset_inputs and (dataset_inputs != parser.shape.inputs):
            raise ValueError(
                f"Model has {parser.shape.inputs} input nodes. ({dataset_inputs} given by the dataset)"
            )
        strategy = OnePlusLambda(_lambda, factory, instance_method,
                                 mutation_method, fitness)
        model = ModelCGP(generations, strategy, parser)
        if callbacks:
            for callback in callbacks:
                callback.set_parser(parser)
                model.attach(callback)
        return model


ENDPOINT_DEFAULT_INSTANCE_SEGMENTATION = EndpointWatershed()
BUNDLE_DEFAULT_INSTANCE_SEGMENTATION = BUNDLE_OPENCV
STACKER_DEFAULT_INSTANCE_SEGMENTATION = MeanKartezioStackerForWatershed()


def create_instance_segmentation_model(
    generations,
    _lambda,
    endpoint=ENDPOINT_DEFAULT_INSTANCE_SEGMENTATION,
    bundle=BUNDLE_DEFAULT_INSTANCE_SEGMENTATION,
    inputs=3,
    nodes=30,
    outputs=2,
    arity=2,
    parameters=2,
    series_mode=False,
    series_stacker=STACKER_DEFAULT_INSTANCE_SEGMENTATION,
    instance_method="random",
    mutation_method="classic",
    node_mutation_rate=0.15,
    output_mutation_rate=0.2,
    use_goldman=True,
    fitness="AP",
    callbacks=None,
    dataset_inputs=None,
):
    builder = ModelBuilder()
    builder.create(
        endpoint,
        bundle,
        inputs,
        nodes,
        outputs,
        arity,
        parameters,
        series_mode=series_mode,
        series_stacker=series_stacker,
    )
    builder.set_instance_method(instance_method)
    builder.set_mutation_method(
        mutation_method,
        node_mutation_rate,
        output_mutation_rate,
        use_goldman=use_goldman,
    )
    builder.set_fitness(fitness)
    model = builder.compile(generations,
                            _lambda,
                            callbacks=callbacks,
                            dataset_inputs=dataset_inputs)
    return model


ENDPOINT_DEFAULT_SEGMENTATION = EndpointThreshold(threshold=4)
BUNDLE_DEFAULT_SEGMENTATION = BUNDLE_OPENCV
STACKER_DEFAULT_SEGMENTATION = StackerMean()


class ModelML(ABC):

    @abstractmethod
    def fit(self, x: List, y: List):
        pass

    @abstractmethod
    def evaluate(self, x: List, y: List):
        pass

    @abstractmethod
    def predict(self, x: List):
        pass


class ModelGA(ABC):

    def __init__(self, strategy, generations):
        self.strategy = strategy
        self.current_generation = 0
        self.generations = generations

    def fit(self, x: List, y: List):
        pass

    def initialization(self):
        self.strategy.initialization()

    def is_satisfying(self):
        end_of_generations = self.current_generation >= self.generations
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


class ModelCGP(ModelML, Observable):

    def __init__(self, generations, strategy, parser):
        super().__init__()
        self.generations = generations
        self.strategy = strategy
        self.parser = parser
        self.callbacks = []

    def fit(
        self,
        x,
        y,
    ):
        genetic_algorithm = ModelGA(self.strategy, self.generations)
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

    def evaluate(self, x, y):
        y_pred, t = self.predict(x)
        return self.strategy.fitness.compute(y, [y_pred])

    def predict(self, x):
        return self.parser.parse(self.strategy.elite, x)

    def save_elite(self, filepath, dataset):
        JsonSaver(dataset, self.parser).save_individual(
            filepath,
            self.strategy.population.history().individuals[0])


@singleton
class TrainingArgs:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def parse(self):
        return self.parser.parse_args()

    def set_arguments(self):
        self.parser.add_argument("output_directory",
                                 help="path to output directory")
        self.parser.add_argument("dataset_directory",
                                 help="path to the dataset directory")
        self.parser.add_argument(
            "--indices",
            help="list of indices to select among dataset for the training",
            nargs="+",
            type=int,
            default=None,
        )
        self.parser.add_argument(
            "--dataset_filename",
            help=f"name of the dataset file, default is {CSV_DATASET}",
            default=CSV_DATASET,
        )
        self.parser.add_argument(
            "--genome",
            help="initial genome to instantiate population",
            default=None)
        self.parser.add_argument("--generations",
                                 help="Number of generations",
                                 default=100,
                                 type=int)


kartezio_parser = TrainingArgs()


def get_args():
    return kartezio_parser.parse()


class KartezioTraining:

    def __init__(self,
                 model: ModelCGP,
                 reformat_x=None,
                 frequency=1,
                 preview=False):
        self.args = get_args()
        self.model = model
        self.dataset = read_dataset(
            self.args.dataset_directory,
            counting=True,
            preview=preview,
            indices=self.args.indices,
        )
        if frequency < 1:
            frequency = 1
        self.callbacks = [
            CallbackVerbose(frequency=frequency),
            CallbackSave(self.args.output_directory,
                         self.dataset,
                         frequency=frequency),
        ]
        self.reformat_x = reformat_x

    def run(self):
        train_x, train_y = self.dataset.train_xy
        if self.reformat_x:
            train_x = self.reformat_x(train_x)
        if self.callbacks:
            for callback in self.callbacks:
                callback.set_parser(self.model.parser)
                self.model.attach(callback)
        elite, history = self.model.fit(train_x, train_y)
        return elite


def train_model(
    model,
    dataset,
    output_directory,
    preprocessing=None,
    callbacks="default",
    callback_frequency=1,
    pack=True,
):
    if callbacks == "default":
        verbose = CallbackVerbose(frequency=callback_frequency)
        save = CallbackSave(output_directory,
                            dataset,
                            frequency=callback_frequency)
        callbacks = [verbose, save]
        workdir = str(save.workdir._path)
        print(f"Files will be saved under {workdir}.")
    if callbacks:
        for callback in callbacks:
            callback.set_parser(model.parser)
            model.attach(callback)
    train_x, train_y = dataset.train_xy
    if preprocessing:
        train_x = preprocessing.call(train_x)
    res = model.fit(train_x, train_y)
    if pack:
        pack_one_directory(workdir)
    return res


model = create_instance_segmentation_model(
    generations=10,
    _lambda=5,
    inputs=3,
    outputs=2,
    endpoint=EndpointWatershed(),
)
elite, _ = train_model(model, read_dataset("dataset"), ".", preprocessing=None)
