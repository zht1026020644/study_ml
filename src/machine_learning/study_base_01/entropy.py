import numpy as np
import matplotlib.pyplot as plt

eps = 1e-4
p = np.linspace(eps, 1 - eps, 100)
h = -(1 - p) * np.log(1 - p) - p * np.log(p)
plt.plot(p, h, 'r-', lw=3)
plt.show()
