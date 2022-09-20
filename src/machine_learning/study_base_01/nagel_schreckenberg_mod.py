import numpy as np
import matplotlib.pyplot as plt

'''
????????
'''

path = 5000  # ??????
n = 100  # ???????
v0 = 5  # ??????
p = 0.3  # ???????
Times = 3000

np.random.seed(0)
x = np.random.rand(n) * path
x.sort()
v = np.tile([v0], n).astype(np.float64)

plt.figure(figsize=(10, 8), facecolor='w')
for t in range(Times):
    plt.scatter(x, [t] * n, s=1, c='k')
    for i in range(n):
        if x[(i + 1) % n] > x[i]:
            d = x[(i + 1) % n] - x[i]
        else:
            d = path - x[i] + x[(i + 1) % n]
        if v[i] < d:
            if np.random.rand() > p:
                v[i] += 1
            else:
                v[i] -= 1
        else:
            v[i] = d - 1
    v = v.clip(0, 150)
    x += v
    x = x % path
plt.xlim(0, path)
plt.ylim(0, Times)
plt.xlabel(u'????', fontsize=16)
plt.ylabel(u'????', fontsize=16)
plt.title(u'????????', fontsize=16)
plt.tight_layout(pad=2)
plt.show()
