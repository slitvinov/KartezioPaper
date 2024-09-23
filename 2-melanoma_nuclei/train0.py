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
from kartezio.endpoint import EndpointWatershed
from kartezio.image.bundle import BUNDLE_OPENCV
from kartezio.stacker import MeanKartezioStackerForWatershed
from kartezio.model.components import (
    GenomeFactory,
    GenomeShape,
    KartezioBundle,
    KartezioEndpoint,
    KartezioParser,
    KartezioStacker,
    ParserChain,
)
from kartezio.model.evolution import KartezioFitness, KartezioMutation
from kartezio.model.registry import registry
from kartezio.mutation import GoldmanWrapper, MutationAllRandom
from kartezio.stacker import StackerMean
from kartezio.model.evolution import KartezioES
from kartezio.model.evolution import KartezioPopulation
from kartezio.callback import CallbackSave, CallbackVerbose
from kartezio.enums import CSV_DATASET
from kartezio.utils.io import pack_one_directory
from kartezio.enums import CSV_DATASET, DIR_PREVIEW, JSON_META
from kartezio.callback import Event
from kartezio.export import GenomeToPython
from kartezio.model.helpers import Observable
from kartezio.utils.io import JsonSaver

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
            self._notify(genetic_algorithm.current_generation, Event.START_STEP)
            genetic_algorithm.selection()
            genetic_algorithm.reproduction()
            genetic_algorithm.mutation()
            y_pred = self.parser.parse_population(self.strategy.population, x)
            genetic_algorithm.evaluation(y, y_pred)
            genetic_algorithm.next()
            self._notify(genetic_algorithm.current_generation, Event.END_STEP)
        self._notify(genetic_algorithm.current_generation, Event.END_LOOP, force=True)
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
            filepath, self.strategy.population.history().individuals[0]
        )

    def print_python_class(self, class_name):
        python_writer = GenomeToPython(self.parser)
        python_writer.to_python_class(class_name, self.strategy.elite)


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

    def __init__(self, train_set, test_set, name, label_name, inputs, indices=None):
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
            "input": {"type": input_type, "format": input_format},
            "label": {"type": label_type, "format": label_format},
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
        return DataItem([labels], image.shape[:2], len(np.unique(labels)) - 1, image)


@registry.readers.add("image_hsv")
class ImageHSVReader(DataReader):
    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_hsv = bgr2hsv(image_bgr)
        return DataItem(image_split(image_hsv), image_bgr.shape[:2], None, image_bgr)


@registry.readers.add("image_hed")
class ImageHEDReader(DataReader):
    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_hed = bgr2hed(image_bgr)
        return DataItem(image_split(image_hed), image_bgr.shape[:2], None, image_bgr)


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
        return DataItem(
            image_split(image), image.shape[:2], None, visual=rgb2bgr(image)
        )


@registry.readers.add("csv_ellipse")
class CsvEllipseReader(DataReader):
    def _read(self, filepath, shape=None):
        dataframe = pd.read_csv(filepath)
        ellipses = read_ellipses_from_csv(
            dataframe, scale=self.scale, ellipse_scale=1.0
        )
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
                (image_new(channels[0].shape), channels[0], channels[1])
            )
        if len(image.shape) == 4:
            # stack: (z, c, h, w)
            channels = [image[:, i] for i in range(len(image[0]))]
            preview = cv2.merge(
                (
                    channels[0].max(axis=0).astype(np.uint8),
                    channels[1].max(axis=0).astype(np.uint8),
                    image_new(channels[0][0].shape, dtype=np.uint8),
                )
            )
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
        self.input_reader = registry.readers.instantiate(
            input_reader_name, directory=self, scale=self.scale
        )
        self.label_reader = registry.readers.instantiate(
            label_reader_name, directory=self, scale=self.scale
        )

    def read_dataset(
        self, dataset_filename=CSV_DATASET, meta_filename=JSON_META, indices=None
    ):
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
            print(f"Inconsistent size of inputs for this dataset: sizes: {input_sizes}")

        if self.preview:
            for i in range(len(training.x)):
                visual = training.v[i]
                label = training.y[i][0]
                preview = draw_overlay(
                    visual, label.astype(np.uint8), color=[224, 255, 255], alpha=0.5
                )
                self.preview_dir.write(f"train_{i}.png", preview)
            for i in range(len(testing.x)):
                visual = testing.v[i]
                label = testing.y[i][0]
                preview = draw_overlay(
                    visual, label.astype(np.uint8), color=[224, 255, 255], alpha=0.5
                )
                self.preview_dir.write(f"test_{i}.png", preview)
        return Dataset(training, testing, self.name, self.label_name, inputs, indices)

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
    dataset_reader = DatasetReader(dataset_path, counting=counting, preview=preview)
    if reader is not None:
        dataset_reader.add_reader(reader)
    return dataset_reader.read_dataset(
        dataset_filename=filename, meta_filename=meta_filename, indices=indices
    )

class TrainingArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def parse(self):
        return self.parser.parse_args()

    def set_arguments(self):
        self.parser.add_argument("output_directory", help="path to output directory")
        self.parser.add_argument(
            "dataset_directory", help="path to the dataset directory"
        )
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
            "--genome", help="initial genome to instantiate population", default=None
        )
        self.parser.add_argument(
            "--generations", help="Number of generations", default=100, type=int
        )


kartezio_parser = TrainingArgs()


def get_args():
    return kartezio_parser.parse()


class KartezioTraining:
    def __init__(self, model: ModelCGP, reformat_x=None, frequency=1, preview=False):
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
            CallbackSave(self.args.output_directory, self.dataset, frequency=frequency),
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
        save = CallbackSave(output_directory, dataset, frequency=callback_frequency)
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
            self.individuals[i].set_values(
                individuals[i].sequence, float(fitness[i]), float(times[i])
            )

    def get_best_fitness(self):
        return (
            self.individuals[0].fitness["fitness"],
            self.individuals[0].fitness["time"],
        )

    def get_individuals(self):
        return self.individuals.items()


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
    def __init__(self, _lambda, factory, init_method, mutation_method, fitness):
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
            self.population[i] = self.mutation_method.mutate(self.population[i])

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

    def __post_init__(
        self, inputs: int, nodes: int, outputs: int, arity: int, parameters: int
    ):
        self.genome_shape = GenomeShape(inputs, nodes, outputs, arity, parameters)
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
            parser = ParserChain(
                self.genome_shape, self.bundle, series_stacker, self.endpoint
            )
        else:
            parser = KartezioParser(self.genome_shape, self.bundle, self.endpoint)
        self.parser = parser
        self.series_mode = series_mode


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
        self.__context = ModelContext(inputs, nodes, outputs, arity, parameters)
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

    def set_mutation_method(
        self, mutation, node_mutation_rate, output_mutation_rate, use_goldman=True
    ):
        if type(mutation) == str:
            shape = self.__context.genome_shape
            n_nodes = self.__context.bundle.size
            mutation = registry.mutations.instantiate(
                mutation, shape, n_nodes, node_mutation_rate, output_mutation_rate
            )
        if use_goldman:
            parser = self.__context.parser
            mutation = GoldmanWrapper(mutation, parser)
        self.__context.set_mutation_method(mutation)

    def set_fitness(self, fitness):
        if type(fitness) == str:
            fitness = registry.fitness.instantiate(fitness)
        self.__context.set_fitness(fitness)

    def compile(self, generations, _lambda, callbacks=None, dataset_inputs=None):
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
                raise ValueError(f"Stacker {parser.stacker} has not been properly set.")

            if parser.stacker.arity != parser.shape.outputs:
                raise ValueError(
                    f"Stacker [{parser.stacker.name}] requires {parser.stacker.arity} output nodes. ({parser.shape.outputs} given)"
                )

        if not isinstance(fitness, KartezioFitness):
            raise ValueError(f"Fitness {fitness} has not been properly set.")

        if not isinstance(mutation_method, KartezioMutation):
            raise ValueError(f"Mutation {mutation_method} has not been properly set.")

        if dataset_inputs and (dataset_inputs != parser.shape.inputs):
            raise ValueError(
                f"Model has {parser.shape.inputs} input nodes. ({dataset_inputs} given by the dataset)"
            )

        strategy = OnePlusLambda(
            _lambda, factory, instance_method, mutation_method, fitness
        )
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
    model = builder.compile(
        generations, _lambda, callbacks=callbacks, dataset_inputs=dataset_inputs
    )
    return model


model = create_instance_segmentation_model(
    generations=10,
    _lambda=5,
    inputs=3,
    outputs=2,
    endpoint=EndpointWatershed(),
)
elite, _ = train_model(model, read_dataset("dataset"), ".")
