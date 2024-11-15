import array
import copy


def forward(a):
    b = copy.copy(a)
    n = len(a) // 2
    while True:
        for i in range(n):
            b[i] = (a[2 * i + 1] + a[2 * i]) / 2
            b[i + n] = a[2 * i] - b[i]
        n //= 2
        if n == 0:
            return b
        a = copy.copy(b)


def backward(a):
    m = len(a)
    n = 1
    b = copy.copy(a)
    while True:
        for i in range(n):
            b[2 * i] = a[i] + a[n + i]
            b[2 * i + 1] = a[i] - a[n + i]
        n *= 2
        if n == m:
            return b
        a = copy.copy(b)


def threshold(a, x):
    n = len(a)
    b = copy.copy(a)
    for i in range(n):
        if abs(b[i]) <= x:
            b[i] = 0
    return b


a = array.array("d", [56, 40, 8, 24, 48, 48, 40, 16])
b = forward(a)

print(backward(threshold(b, 4)))
print(a)
