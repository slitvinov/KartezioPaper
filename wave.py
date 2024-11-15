import matplotlib.pyplot as plt
import sys


def u0(x):
    a = abs(x)
    return 1 - a if a <= 1 else 0


def u1(t, x):
    return u0(t - x)


def forward_forward():
    u[m] = v[m] - lmb * (v[m + 1] - v[m - 1]) / 2


def lax_friedrichs():
    u[m] = (v[m - 1] + v[m + 1]) / 2 - lmb * (v[m + 1] - v[m - 1]) / 2


# scheme, lmb  = forward_forward, 0.1
scheme, lmb = lax_friedrichs, 0.8

tend = 1.6
lmb = 0.8
M = 50
lo, hi = -2, 3
h = (hi - lo) / M
k = lmb * h
x = [lo + i * h for i in range(M + 1)]
t = 0
v = [u0(x) for x in x]
u = [None] * (M + 1)
while t < tend:
    u[0] = 0
    for m in range(1, M):
        scheme()
    u[M] = u[M - 1]
    t += k
    u, v = v, u
plt.plot(x,
         v,
         'ko-',
         x, [u1(t, x) for x in x],
         '-k',
         x, [u0(x) for x in x],
         '--k',
         mfc='none')
path = "wave.png"
plt.savefig(path)
sys.stderr.write("wave.py: %s\n" % path)
