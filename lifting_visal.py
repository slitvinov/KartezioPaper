from random import randrange
import multiprocessing
import numpy as np
import random
import os
import itertools


# Forward Lifting Haar Wavelet Transform
def lifting_haar_wavelet_transform(signal):
    # Ensure the signal length is even
    if len(signal) % 2 != 0:
        signal = np.append(signal, 0)

    N = len(signal)

    # Split the signal into even and odd indices
    even = signal[::2]
    odd = signal[1::2]

    # Prediction step (Lift the odd coefficients)
    odd -= even

    # Update step (Update even coefficients)
    even += 0.5 * odd

    return even, odd


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
        return x + y


class Minus:
    arity = 2
    args = 0

    def call(self, inp, args):
        x, y = inp
        return x - y


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
            z[i] = x[i]
            z[i + N // 2] = y[i]
        return z


Nodes = {
    "Odd": Odd,
    "Even": Even,
    "Merge": Merge,
    "Plus": Plus,
    "Minus": Minus,
    "U": U,
    #    "P": P,
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
    for count in range(2):
        x = [np.floor(np.random.uniform(1, 100, size=(8, )))]
        coarse, detail = lifting_haar_wavelet_transform(np.copy(x[0]))
        y = [np.append(coarse, detail)]
        values = {i: x[i] for i in range(g.i)}
        for n in topo:
            arity = g.nodes[gen[n, 0]].arity
            inputs = [values[i] for i in gen[n, 1:1 + arity]]
            params = gen[n, 1 + g.a:]
            values[n] = g.nodes[gen[n, 0]].call(inputs, params)
        y_pred = [values[j] for j in gen[g.i + g.n:, 1]]
        Cost += diff(y, y_pred)
    if len(topo) == 0:
        Cost_final = Cost / len(g.y) / 2. + len(topo)
    else:
        Cost_final = Cost / len(g.y) / 2. + len(topo) + np.max(topo)
    return Cost_final


def diff(a, b):
    a, = a
    b, = b
    diff = a - b
    return np.mean(diff**2)


class G:
    pass


random.seed(2)
np.random.seed(2)
g = G()
x0 = np.array([56, 40, 8, 24, 48, 48, 40, 16], dtype=float)
N = len(x0)
y0 = np.array([48, 16, 48, 28, -16, 16, 0, -24], dtype=float)

g.x = [[x0]]
g.y = [[y0]]
g.max_val = 256
g.lmb = 20000
max_generation = 130
g.nodes = [cls() for cls in Nodes.values()]
# input, maximum node, otuput, arity, parameters
g.i = 1
g.n = 12
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
        costs = pool.map(fun, genes)
    idx = sorted((cost, i) for i, cost in enumerate(costs))
    os.makedirs(f"{generation:08}", exist_ok=True)
    with open(os.path.join(f"{generation:08}", "cost"), "w") as f:
        for j, (cost, i) in enumerate(idx[:500]):
            graph(genes[i], os.path.join(f"{generation:08}", f"{j:08}.gv"))
            f.write(f"{cost:.16e}\n")
    print(f"{generation:08} {costs[idx[0][1]]:8.2e} {costs[idx[1][1]]:8.2e}")
    i = np.argmin(costs)
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
