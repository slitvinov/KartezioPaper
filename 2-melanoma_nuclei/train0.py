from kartezio.endpoint import (
    EndpointEllipse,
    EndpointHoughCircle,
    EndpointLabels,
    EndpointWatershed,
    LocalMaxWatershed,
)
import argparse
from typing import List
from abc import ABC, abstractmethod
from numena.io.drive import Directory

from kartezio.callback import CallbackSave, CallbackVerbose
from kartezio.callback import Event
from kartezio.dataset import read_dataset
from kartezio.endpoint import EndpointThreshold
from kartezio.enums import CSV_DATASET
from kartezio.export import GenomeToPython
from kartezio.image.bundle import BUNDLE_OPENCV
from kartezio.model.helpers import Observable
from kartezio.model.helpers import singleton
from kartezio.preprocessing import TransformToHED, TransformToHSV
from kartezio.stacker import MeanKartezioStackerForWatershed
from kartezio.stacker import StackerMean
from kartezio.utils.io import JsonSaver
from kartezio.utils.io import pack_one_directory

from dataclasses import InitVar, dataclass, field

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
from kartezio.strategy import OnePlusLambda


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


@singleton
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

model = create_instance_segmentation_model(
    generations=20,
    _lambda=5,
    inputs=3,
    outputs=2,
    endpoint=EndpointWatershed(),
)
elite, _ = train_model(model,
                       read_dataset("dataset"),
                       ".",
                       preprocessing=None)
