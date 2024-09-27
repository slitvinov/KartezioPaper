from random import randrange
import multiprocessing
import numpy as np
import os
import random
import functools


class Forward_X:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        y = np.empty((N, N))
        for j in range(N):
            for i in range(N - 1):
                y[i, j] = x[i + 1, j] - x[i, j]
            y[N - 1, j] = x[0, j] - x[N - 1, j]
        return y


class Backward_X:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        y = np.empty((N, N))
        for j in range(N):
            for i in range(1, N):
                y[i, j] = x[i, j] - x[i - 1, j]
            y[0, j] = x[0, j] - x[N - 1, j]
        return y


class Forward_Y:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        y = np.empty((N, N))
        for i in range(N):
            for j in range(N - 1):
                y[i, j] = x[i, j + 1] - x[i, j]
            y[i, N - 1] = x[i, 0] - x[i, N - 1]
        return y


class Backward_Y:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        y = np.empty((N, N))
        for i in range(N):
            for j in range(1, N):
                y[i, j] = x[i, j] - x[i, j - 1]
            y[i, 0] = x[i, 0] - x[i, N - 1]
        return y


class Plus:
    arity = 2
    args = 0

    def call(self, inp, args):
        x, y = inp
        return x + y


Nodes = {
    "Backward_X": Backward_X,
    "Backward_Y": Backward_Y,
    "Forward_X": Forward_X,
    "Forward_Y": Forward_Y,
    "Plus": Plus,
}


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
        values = {i: x[i] for i in range(g.i)}
        for n in topo:
            arity = g.nodes[gen[n, 0]].arity
            inputs = [values[i] for i in gen[n, 1:1 + arity]]
            params = gen[n, 1 + g.a:]
            values[n] = g.nodes[gen[n, 0]].call(inputs, params)
        y_pred = [values[j] for j in gen[g.i + g.n:, 1]]
        Cost += diff(y, y_pred)
    return Cost / len(g.y)


def diff(a, b):
    diff = np.subtract(a[0][1:-1, 1:-1], b[0][1:-1, 1:-1])
    return np.mean(diff**2)


class G:
    pass


random.seed(2)
np.random.seed(2)
g = G()

N = 10
Y, X = np.meshgrid(range(N), range(N))
x0 = Y**2 + 2 * X**2
y0 = np.full((N, N), 2 + 2 * 2)

g.x = [[x0]]
g.y = [[y0]]
g.max_val = 256
g.lmb = 20
max_generation = 2000
g.nodes = [cls() for cls in Nodes.values()]
# input, maximum node, otuput, arity, parameters
g.i = 1
g.n = 10
g.o = 1
g.a = 2
g.p = 0
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
    with multiprocessing.Pool() as pool:
        cost = pool.map(fun, genes)
    i = np.argmin(cost)
    if generation % 10 == 0:
        graph(genes[i], f"diff.{generation:08}.gv")
        print(f"{generation:08} {cost[i]:.16e}")
    if generation == max_generation:
        break
    generation += 1
    elite = genes[0] = genes[i]
    topo = stopo(elite)
    for i in range(1, g.lmb + 1):
        while True:
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
            if stopo(gen) != topo:
                break
