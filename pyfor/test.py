import importlib.machinery
import numpy as np
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()


pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/tiles/PC_076.las")

pc_grid = pc.grid(1)

interp = pc_grid.interpolate("z", "min")

plt.imshow(interp.T)
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()
