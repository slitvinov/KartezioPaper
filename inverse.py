from random import randrange
import multiprocessing
import numpy as np
import random


class Even:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        y = np.zeros(N)
        for i in range(N // 2):
            y[i] = x[2 * i]
        return y


class Odd:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        y = np.zeros(N)
        for i in range(N // 2):
            y[i] = x[2 * i + 1]
        return y


class Plus:
    arity = 2
    args = 0

    def call(self, inp, args):
        x, y = inp
        z = np.zeros(N)
        for i in range(N):
            z[i] = x[i] + y[i]
        return z


class Minus:
    arity = 2
    args = 0

    def call(self, inp, args):
        x, y = inp
        z = np.zeros(N)
        for i in range(N // 2):
            z[i] = x[i] - y[i]
        return z


class P:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        return x


class U:
    arity = 1
    args = 0

    def call(self, inp, args):
        x, = inp
        return x / 2


class Merge:
    arity = 2
    args = 0

    def call(self, inp, args):
        x, y = inp
        z = np.empty(N)
        for i in range(N // 2):
            z[2 * i] = x[i]
            z[2 * i + 1] = y[i]
        return z


Nodes = {
    "Odd": Odd,
    "Even": Even,
    "Merge": Merge,
    "Plus": Plus,
    "Minus": Minus,
    "U": U,
    "P": P,
}
Names = dict(enumerate(Nodes.keys()))


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
    with open(path, "w") as f:
        f.write("digraph {\n")
        for j in range(g.i):
            f.write(f"  {j} [label = i{j}]\n")
        for n in stopo(gen):
            arity = g.nodes[gen[n, 0]].arity
            args = g.nodes[gen[n, 0]].args
            f.write(f'  {n} [label = "{Names[gen[n, 0]]}')
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
    return Cost / len(g.y) + len(topo)


def diff(a, b):
    a, = a
    b, = b
    diff = a - b
    return np.mean(diff**2)


def good(gen):
    j = gen[g.i + g.n + 0, 1]
    if j > g.i and Names[gen[j, 0]] == "Merge":
        return True
    return False


def init():
    genes = [
        np.empty((g.i + g.n + g.o, 1 + g.a + g.p), dtype=np.uint8)
        for i in range(g.lmb + 1)
    ]
    for gen in genes:
        while True:
            for j in range(g.n):
                gen[g.i + j, 0] = randrange(len(g.nodes))
                for k in range(g.a):
                    gen[g.i + j, 1 + k] = randrange(g.i + j)
                for k in range(g.p):
                    gen[g.i + j, 1 + g.a + k] = randrange(g.max_val)
            for j in range(g.o):
                gen[g.i + g.n + j, 1] = randrange(g.i + g.n)
            if good(gen):
                break
    return genes


def mutate(i, genes):
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
            if stopo(gen) != topo and good(gen):
                break


class G:
    pass


random.seed(2)
np.random.seed(2)
g = G()

x0 = np.array([56, 40, 8, 24, 48, 48, 40, 16], dtype=float)
N = len(x0)
y0 = np.array([48, -16, 16, 16, 48, 0, 28, -24], dtype=float)

g.x = [[x0]]
g.y = [[y0]]
g.max_val = 256
g.lmb = 5000
max_generation = 100
g.nodes = [cls() for cls in Nodes.values()]
# input, maximum node, otuput, arity, parameters
g.i = 1
g.n = 9
g.o = 1
g.a = 2
g.p = 0

genes = init()
generation = 0
n_mutations = 30 * g.n * (1 + g.a + g.p) // 100
while True:
    with multiprocessing.Pool() as pool:
        cost = pool.map(fun, genes)
    i = np.argmin(cost)
    if generation % 10 == 0:
        graph(genes[i], f"inverse.{generation:08}.gv")
        print(f"{generation:08} {cost[i]:.16e} {max(cost):.16e}")
    if generation == max_generation:
        break
    generation += 1
    mutate(i, genes)
