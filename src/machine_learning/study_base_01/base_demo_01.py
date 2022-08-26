import numpy as np
import matplotlib.pyplot as plt
data = np.random.rand(100, 2)
x = data[:, 0]
y = data[:, 1]
idx = x**2 + y**2 <1
print(idx)
plt.plot(x[idx], y[idx], 'go', markersize=1)


plt.show()