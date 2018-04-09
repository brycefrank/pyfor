import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)

pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/tiles/PC_076.las")
pc_grid = pc.grid(0.5)

A = pc_grid.interpolate("min", "z")

from scipy.ndimage.morphology import grey_opening
dem = pyfor.filter.zhang(A, 5, 3, 0.5, 0.5, pc_grid)
plt.matshow(dem)
plt.show()