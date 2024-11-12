import matplotlib.pyplot as plt


def u0(x):
    a = abs(x)
    return 1 - a if a <= 1 else 0


def u1(t, x):
    return u0(t - x)


tend = 1.6
lmb = 0.1
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
        # forward-time, forward-space
        u[m] = v[m] - lmb * (v[m + 1] - v[m - 1]) / 2

        # Lax-Friedrichs
        # u[m] = (v[m - 1] + v[m + 1]) / 2 - lmb * (v[m + 1] - v[m - 1]) / 2
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
plt.savefig("wave.png")
