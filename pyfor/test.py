import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/pyfor/pyfor/__init__.py').load_module()
reload(pyfor)

pc = pyfor.cloud.Cloud("/home/bryce/Programming/pyfor/samples/data/small_test.las")

pc_grid = pc.grid(0.3)
pc_grid.plot("max", "count")


A = pc_grid._interpolate("min", "z")
pc_grid.ground_filter()


empty = pc_grid.empty_cells
empty_x, empty_y = empty[:,0], empty[:,1]
A[empty_y, empty_x] = np.nan
B = np.where(flg != 0, A, np.nan)


X, Y = np.mgrid[0:pc_grid.n, 0:pc_grid.m]
# This is where the data is

C = np.where(np.isfinite(B) == True)
vals = B[C[0], C[1]]
from scipy.interpolate import griddata
dem_array = griddata(np.stack((C[0], C[1]), axis = 1), vals, (X, Y))
plt.matshow(dem_array)
plt.show()