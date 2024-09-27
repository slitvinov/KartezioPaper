from nodes import Nodes
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
from random import randrange

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


def stopo(gen):
    q = {x for x in gen[g.i + g.n:, 1]}
    topo = set()
    while q:
        n = q.pop()
        if n >= g.i:
            topo.add(n)
            arity = g.nodes[gen[n, 0]].arity
            adj = gen[n, 1:1 + arity]
            q.update(adj)
    return sorted(topo)


def graph(gen, path):
    names = dict(enumerate(Nodes.keys()))
    with open(path, "w") as f:
        f.write("digraph {\n")
        for j in range(g.i):
            f.write(f"  {j} [label = i{j}]\n")
        for n in stopo(gen):
            arity = g.nodes[gen[n, 0]].arity
            args = g.nodes[gen[n, 0]].args
            f.write(f'  {n} [label = "{names[gen[n, 0]]}')
            for j in range(args):
                f.write(f", {gen[n, 1 + g.a + j]}")
            f.write('"]\n')
            for j in range(arity):
                f.write(f"  {gen[n, 1 + j]} -> {n}\n")
        for j in range(g.o):
            f.write(f"  {g.i + g.n + j} [label = o{j}]\n")
            f.write(f"  {gen[g.i + g.n + j, 1]} -> {g.i + g.n + j}\n")
        f.write("}\n")


def fun(gen):
    topo = stopo(gen)
    Cost = 0
    for x, y in zip(g.x, g.y):
        values = {i: x[i].copy() for i in range(g.i)}
        for n in topo:
            arity = g.nodes[gen[n, 0]].arity
            inputs = [values[i] for i in gen[n, 1:1 + arity]]
            params = gen[n, 1 + g.a:]
            values[n] = g.nodes[gen[n, 0]].call(inputs, params)
        y_pred = [values[j] for j in gen[g.i + g.n:, 1]]
        *rest, y_pred = g.wt.apply(y_pred[0],
                                   markers=y_pred[1],
                                   mask=y_pred[0] > 0)
        Cost += diff(y, y_pred)
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


random.seed(2)
np.random.seed(2)
g = G()
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
g.max_val = 256
g.lmb = 5
max_generation = 100
g.wt = WatershedSkimage(use_dt=False, markers_distance=21, markers_area=None)
g.nodes = [cls() for cls in Nodes.values()]
# input, maximum node, otuput, arity, parameters
g.i = 3
g.n = 30
g.o = 2
g.a = 2
g.p = 2
genes = [
    np.zeros((g.i + g.n + g.o, 1 + g.a + g.p), dtype=np.uint8)
    for i in range(g.lmb + 1)
]
for gen in genes:
    for j in range(g.n):
        gen[g.i + j, 0] = randrange(len(g.nodes))
        for k in range(g.a):
            gen[g.i + j, 1 + k] = randrange(g.i + j)
        for k in range(g.p):
            gen[g.i + j, 1 + g.a + k] = randrange(g.max_val)
    for j in range(g.o):
        gen[g.i + g.n + j, 1] = randrange(g.i + g.n)
generation = 0
n_mutations = 15 * g.n * (1 + g.a + g.p) // 100
while True:
    cost = [fun(gen) for gen in genes]
    i = np.argmin(cost)
    graph(genes[i], f"main.{generation:08}.gv")
    print(f"{generation:08} {cost[i]:.16e}")
    if generation == max_generation:
        break
    generation += 1
    elite = genes[0] = genes[i]
    for i in range(1, g.lmb + 1):
        gen = genes[i] = elite.copy()
        for m in range(n_mutations):
            j = randrange(g.n)
            k = randrange(1 + g.a + g.p)
            if k == 0:
                gen[g.i + j, 0] = randrange(len(g.nodes))
            elif k <= g.a:
                gen[g.i + j, k] = randrange(g.i + j)
            else:
                gen[g.i + j, k] = randrange(g.max_val)
        for k in range(g.o):
            if random.random() < 0.2:
                gen[g.i + g.n + k, 1] = randrange(g.i + g.n)
