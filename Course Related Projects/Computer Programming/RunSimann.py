import numpy as np
from matplotlib import pyplot as plt

from ClassFinal import Grid
from SimAnnFinal import simann

n = 100
student_id = "3232295"

grid = Grid(n, student_id)  # the landscape is a deterministic function of the ID
best_grid, best_c = simann(grid, beta0=0.01, beta1=50, anneal_steps=50,
                           mcmc_steps=n, seed=3232295, plot=True)

f = grid.f_values
gx, gy = np.unravel_index(np.argmin(f), f.shape)
print(f"best found:     f = {best_c:.4f} at ({best_grid.position_x}, {best_grid.position_y})")
print(f"global minimum: f = {f[gx, gy]:.4f} at ({gx}, {gy})")
plt.show()
