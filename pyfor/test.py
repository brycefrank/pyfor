import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)



pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")
pc_grid = pc.grid(0.5)

import matplotlib.pyplot as plt


A = pc_grid.interpolate("min", "z")

plt.matshow(A)
flag = pyfor.filter.zhang(A, 3, 3, 1, 0.5)
print(sum(flag))
test_z = A[0,:]

plt.matshow(flag)

# This gets rid of the interpolated cells
empty_y, empty_x = pc_grid.empty_cells[:,0].astype(int), pc_grid.empty_cells[:,1].astype(int)

A[empty_x - 1, empty_y - 1] = np.nan

B = np.where(flag == 0, A, np.nan)
plt.matshow(B)


