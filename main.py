from nodes import registry
from numba import jit
from numena.image.basics import image_new
from numena.image.basics import image_split
from numena.image.drawing import fill_polygons_as_labels
from numena.image.morphology import WatershedSkimage
from numena.io.image import imread_color
from numena.io.imagej import read_polygons_from_roi
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
import random

DATA = [
    ["dataset/03.png", "dataset/03.zip"],  #
    ["dataset/07.png", "dataset/07.zip"],
    ["dataset/09.png", "dataset/09.zip"],
    ["dataset/10.png", "dataset/10.zip"],
    ["dataset/11.png", "dataset/11.zip"],
    ["dataset/14.png", "dataset/14.zip"],
    ["dataset/17.png", "dataset/17.zip"],
    ["dataset/19.png", "dataset/19.zip"],
    ["dataset/21.png", "dataset/21.zip"],
    ["dataset/23.png", "dataset/23.zip"],
    ["dataset/24.png", "dataset/24.zip"],
]


def _parse_one_graph(gen, source):
    next_indices = {source}
    output_tree = {source}
    while next_indices:
        next_index = next_indices.pop()
        if next_index < g.i:
            continue
        idx = next_index - g.i
        function_index = gen[g.i + idx, 0]
        arity = g.nodes[function_index].arity
        next_connections = set(gen[g.i + idx, 1:1 + arity])
        next_indices = next_indices.union(next_connections)
        output_tree = output_tree.union(next_connections)
    return sorted(output_tree)


def _parse_one(gen, graphs, x):
    output_map = {i: x[i].copy() for i in range(g.i)}
    for graph in graphs:
        for node in graph:
            if node < g.i:
                continue
            idx = node - g.i
            function_index = gen[g.i + idx, 0]
            arity = g.nodes[function_index].arity
            connections = gen[g.i + idx, 1:1 + arity]
            inputs = [output_map[c] for c in connections]
            p = gen[g.i + idx, g.para_idx:]
            output_map[node] = g.nodes[function_index].call(inputs, p)
    return [output_map[output_gene] for output_gene in gen[g.out_idx:, 1]]


def cost(gen):
    graphs = [_parse_one_graph(gen, output) for output in gen[g.out_idx:, 1]]
    Cost = 0
    for xi, yi in zip(g.x, g.y):
        y_pred = _parse_one(gen, graphs, xi)
        mask, markers, y_pred = g.wt.apply(y_pred[0],
                                           markers=y_pred[1],
                                           mask=y_pred[0] > 0)
        Cost += diff(yi, y_pred)
    return Cost / len(g.y)


@jit(nopython=True)
def label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def intersection_over_union(masks_true, masks_pred):
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def diff(y_true, y_pred):
    th = 0.5
    n_true = np.max(y_true)
    n_pred = np.max(y_pred)
    tp = 0
    if n_pred > 0:
        iou = intersection_over_union(y_true, y_pred)[1:, 1:]
        n_min = min(iou.shape[0], iou.shape[1])
        costs = -(iou >= th).astype(float) - iou / (2 * n_min)
        true_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = iou[true_ind, pred_ind] >= th
        tp = match_ok.sum()
    fp = n_pred - tp
    fn = n_true - tp
    if tp == 0:
        return 0 if n_true == 0 else 1
    else:
        return (fp + fn) / (tp + fp + fn)


class G:
    pass


random.seed(1)
np.random.seed(1)
g = G()
g.max_val = 256
g._lambda = 5
g.generations = 10
g.wt = WatershedSkimage(use_dt=False, markers_distance=21, markers_area=None)
g.nodes = [cls() for cls in registry.nodes.components]
g.i = 3
g.n = 30
g.outputs = 2
g.arity = 2
g.parameters = 2
g.out_idx = g.i + g.n
g.para_idx = 1 + g.arity
g.w = 1 + g.arity + g.parameters
g.h = g.i + g.n + g.outputs
g.n_mutations = 15 * g.n * g.w // 100
g.indices = [[i, j] for i in range(g.n) for j in range(g.w)]
g.individuals = [
    np.zeros((g.h, g.w), dtype=np.uint8) for i in range(g._lambda + 1)
]
g.x = []
g.y = []
for sample, label in DATA:
    image = imread_color(sample, rgb=False)
    x, shape = image_split(image), image.shape[:2]
    g.x.append(x)
    label_mask = image_new(shape)
    polygons = read_polygons_from_roi(label)
    fill_polygons_as_labels(label_mask, polygons)
    g.y.append(label_mask)
for gen in g.individuals:
    for j in range(g.n):
        gen[g.i + j, 0] = random.randrange(len(g.nodes))
        gen[g.i + j, 1:g.para_idx] = np.random.randint(g.i + j, size=g.arity)
        gen[g.i + j, g.para_idx:] = np.random.randint(g.max_val,
                                                      size=g.parameters)
    for j in range(g.outputs):
        gen[g.out_idx + j, 1] = random.randrange(g.out_idx)
current_generation = 0
while True:
    g.cost = [cost(gen) for gen in g.individuals]
    i = np.argmin(g.cost)
    print(f"{current_generation:08} {g.cost[i]:.16e}")
    if current_generation == g.generations:
        break
    current_generation += 1
    elite = g.individuals[0] = g.individuals[i]
    for i in range(1, g._lambda + 1):
        gen = g.individuals[i] = elite.copy()
        for idx, j in random.sample(g.indices, g.n_mutations):
            if j == 0:
                gen[g.i + idx, 0] = random.randrange(len(g.nodes))
            elif j <= g.arity:
                gen[g.i + idx,
                    1:g.para_idx][j - 1] = random.randrange(g.i + idx)
            else:
                gen[g.i + idx,
                    g.para_idx:][j - g.arity - 1] = random.randrange(g.max_val)
        for idx in range(g.outputs):
            if random.random() < 0.2:
                gen[g.out_idx + idx, 1] = random.randrange(g.out_idx)
