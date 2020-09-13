import numpy as np

N = 8
d = []
for i in range(N):
    r = np.random.rand(1, 1)[0][0]
    d.append(r)
    print("{:.3f}".format(r))

print()

p = []
# left = 1.0
sumlog = 0.0
for i in range(N):
    """
    r = 0.0
    if i == N - 1:
        r = left
    else:
        while r <= 0 or r >= left:
            r = list(np.random.rand(1, 1))[0][0]
    p.append(r * left)
    left -= r
    """
    sumlog += np.log(d[i])

print(sumlog)

for i in range(N):
    p.append(np.log(d[i]) / sumlog)

# np.random.shuffle(p)

print()

for x in p:
    print("{:.3f}".format(x))

v = [-10.3, -9.4, -11.2, -7.2, +24.5, 12.9, -15.1, -5.2]
for i in v:
    i /= 100.0


sum_influence = 0.0
influence = []
for i in range(N):
    influence.append(p[i] * v[i])
    sum_influence += p[i] * v[i]

print()

for x in influence:
    print("{:.3f}".format(x))


print()

print(sum_influence)
