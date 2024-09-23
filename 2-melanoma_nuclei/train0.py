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
from kartezio.endpoint import EndpointWatershed
from kartezio.image.bundle import BUNDLE_OPENCV
from kartezio.stacker import MeanKartezioStackerForWatershed
from kartezio.model.base import ModelCGP
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
