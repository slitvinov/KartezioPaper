import numpy as np

def example():
    N = 8
    p = 2
    q = 10
    x = [random.randint(-p, p)]
    for i in range(N - 1):
        x.append(x[-1] + random.randint(-p, p))
        p, q = q, p
    return np.array(x, dtype=float)


def forward(a):
    b = np.copy(a)
    n = len(a) // 2
    while True:
        for i in range(n):
            b[i] = (a[2 * i + 1] + a[2 * i]) / 2
            b[i + n] = a[2 * i] - b[i]
        n //= 2
        if n == 0:
            return b
        a = np.copy(b)


def backward(a):
    m = len(a)
    n = 1
    b = np.copy(a)
    while True:
        for i in range(n):
            b[2 * i] = a[i] + a[n + i]
            b[2 * i + 1] = a[i] - a[n + i]
        n *= 2
        if n == m:
            return b
        a = np.copy(b)


def compress(a):
    n = len(a)
    b = np.copy(a)
    i = np.argmin(abs(a[1::2]))
    print(b)
    b[1::2][i] = 0
    print(b)
    exit(0)
    return b


a = np.array([56, 40, 8, 24, 48, 48, 40, 16], dtype=float)
b = forward(a)

print(backward(compress(b)))
print(a)
