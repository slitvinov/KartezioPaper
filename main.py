from numba import jit
from numena.image.basics import image_new
from numena.image.basics import image_split
from numena.image.drawing import fill_polygons_as_labels
from numena.image.morphology import WatershedSkimage
from numena.io.image import imread_color
from numena.io.imagej import read_polygons_from_roi
from numena.io.json import json_read
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
import pandas as pd
import random

from nodes import registry


def _parse_one_graph(genome, graph_source):
    next_indices = graph_source.copy()
    output_tree = graph_source.copy()
    while next_indices:
        next_index = next_indices.pop()
        if next_index < g.inputs:
            continue
        idx = next_index - g.inputs
        function_index = genome[g.inputs + idx, 0]
        arity = g.nodes[function_index].arity
        next_connections = set(genome[g.inputs + idx, 1:1 + arity])
        next_indices = next_indices.union(next_connections)
        output_tree = output_tree.union(next_connections)
    return sorted(list(output_tree))


def parse_to_graphs(genome):
    outputs = genome[g.out_idx:, :]
    graphs_list = [_parse_one_graph(genome, {output[1]}) for output in outputs]
    return graphs_list


def _x_to_output_map(genome, graphs_list, x):
    output_map = {i: x[i].copy() for i in range(g.inputs)}
    for graph in graphs_list:
        for node in graph:
            if node < g.inputs:
                continue
            idx = node - g.inputs
            function_index = genome[g.inputs + idx, 0]
            arity = g.nodes[function_index].arity
            connections = genome[g.inputs + idx, 1:1 + arity]
            inputs = [output_map[c] for c in connections]
            p = genome[g.inputs + idx, g.para_idx:]
            output_map[node] = g.nodes[function_index].call(inputs, p)
    return output_map


def _parse_one(genome, graphs_list, x):
    output_map = _x_to_output_map(genome, graphs_list, x)
    return [
        output_map[output_gene[1]] for output_gene in genome[g.out_idx:, :]
    ]


def parse(genome, x):
    all_y_pred = []
    graphs = parse_to_graphs(genome)
    for xi in x:
        y_pred = _parse_one(genome, graphs, xi)
        mask, markers, y_pred = g.wt.apply(y_pred[0],
                                           markers=y_pred[1],
                                           mask=y_pred[0] > 0)
        all_y_pred.append(y_pred)
    return all_y_pred


def call1(y_true, y_pred):
    scores = []
    for yi_pred in y_pred:
        score = 0.0
        y_size = len(y_true)
        for i in range(y_size):
            score += call0(y_true[i].copy(), yi_pred[i])
        scores.append(score / y_size)
    return scores


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


def call0(y_true, y_pred):
    n_true = np.max(y_true[0])
    n_pred = np.max(y_pred)
    tp = 0
    if n_pred > 0:
        iou = _intersection_over_union(y_true[0], y_pred)[1:, 1:]
        tp = true_positive0(iou)
    fp = n_pred - tp
    fn = n_true - tp
    if tp == 0:
        if n_true == 0:
            return 0.0
        else:
            return 1.0
    else:
        return (fp + fn) / (tp + fp + fn)


def true_positive0(iou):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= g.th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= g.th
    return match_ok.sum()


def mutate_connections(genome, idx, only_one):
    new_connections = np.random.randint(g.inputs + idx, size=g.arity)
    new_value = new_connections[only_one]
    new_connections = genome[g.inputs + idx, 1:g.para_idx]
    new_connections[only_one] = new_value
    genome[g.inputs + idx, 1:g.para_idx] = new_connections


def mutate_parameters1(genome, idx, only_one):
    new_parameters = np.random.randint(g.max_val, size=g.parameters)
    old_parameters = genome[g.inputs + idx, g.para_idx:]
    old_parameters[only_one] = new_parameters[only_one]
    new_parameters = old_parameters.copy()
    genome[g.inputs + idx, g.para_idx:] = new_parameters


def mutate1(genome):
    sampling_indices = np.random.choice(g.sampling_range,
                                        g.n_mutations,
                                        replace=False)
    sampling_indices = g.all_indices[sampling_indices]
    for idx, mutation_parameter_index in sampling_indices:
        if mutation_parameter_index == 0:
            genome[g.inputs + idx, 0] = np.random.randint(len(g.nodes))
        elif mutation_parameter_index <= g.arity:
            connection_idx = mutation_parameter_index - 1
            mutate_connections(genome, idx, only_one=connection_idx)
        else:
            parameter_idx = mutation_parameter_index - g.arity - 1
            mutate_parameters1(genome, idx, only_one=parameter_idx)
    for idx in range(g.outputs):
        if random.random() < 0.2:
            genome[g.out_idx + idx, 1] = np.random.randint(g.out_idx, size=1)
    return genome


class G:
    pass


random.seed(1)
np.random.seed(1)
g = G()
g.max_val = 256
g._lambda = 5
g.generations = 10
g.wt = WatershedSkimage(use_dt=False, markers_distance=21, markers_area=None)
g.nodes = [
    registry.nodes.instantiate(name) for name in registry.nodes.list().keys()
]
g.inputs = 3
g.n = 30
g.outputs = 2
g.arity = 2
g.parameters = 2
g.out_idx = g.inputs + g.n
g.para_idx = 1 + g.arity
g.w = 1 + g.arity + g.parameters
g.h = g.inputs + g.n + g.outputs
g.th = 0.5
g.n_mutations = int(np.floor(0.15 * g.n * g.w))
g.all_indices = np.indices((g.n, g.w))
g.all_indices = np.vstack(
    (g.all_indices[0].ravel(), g.all_indices[1].ravel())).T
g.sampling_range = range(len(g.all_indices))
g.individuals = [None] * (g._lambda + 1)
g.fitness = np.zeros(g._lambda + 1)
meta = json_read("dataset/META.json")
name = meta["name"]
label_name = meta["label_name"]
dataframe = pd.read_csv("dataset/dataset.csv")
dataframe_training = dataframe[dataframe["set"] == "training"]
dataframe_training.reset_index(inplace=True)
x0 = []
y0 = []
for row in dataframe_training.itertuples():
    filepath = os.path.join("dataset", row.input)
    image = imread_color(filepath, rgb=False)
    x, shape = image_split(image), image.shape[:2]
    x0.append(x)
    label_mask = image_new(shape)
    if str(row.label) != "nan":
        filepath = os.path.join("dataset", row.label)
        polygons = read_polygons_from_roi(filepath)
        fill_polygons_as_labels(label_mask, polygons)
    y0.append([label_mask])
for i in range(g._lambda + 1):
    g.individuals[i] = np.zeros((g.h, g.w), dtype=np.uint8)
    for j in range(g.n):
        g.individuals[i][g.inputs + j, 0] = np.random.randint(len(g.nodes))
        mutate_connections(g.individuals[i], j, None)
        new_parameters = np.random.randint(g.max_val, size=g.parameters)
        g.individuals[i][g.inputs + j, g.para_idx:] = new_parameters
    for j in range(g.outputs):
        g.individuals[i][g.out_idx + j, 1] = np.random.randint(g.out_idx,
                                                               size=1)
y_pred = []
for i in range(len(g.individuals)):
    y = parse(g.individuals[i], x0)
    y_pred.append(y)
g.fitness = call1(y0, y_pred)
print(f"{0:08} {g.fitness[0]:.16e}")
current_generation = 0
while current_generation < g.generations:
    i = np.argmin(g.fitness)
    elite = g.individuals[i].copy()
    for i in range(g._lambda + 1):
        g.individuals[i] = elite.copy()
    for i in range(1, g._lambda + 1):
        active_nodes = parse_to_graphs(g.individuals[i])
        while True:
            g.individuals[i] = mutate1(g.individuals[i])
            new_active_nodes = parse_to_graphs(g.individuals[i])
            if active_nodes != new_active_nodes:
                break
    y_pred = []
    for i in range(len(g.individuals)):
        y = parse(g.individuals[i], x0)
        y_pred.append(y)
    g.fitness = call1(y0, y_pred)
    current_generation += 1
    print(f"{current_generation:08} {g.fitness[0]:.16e}")
