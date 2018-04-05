import importlib.machinery
import numpy as np
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()


pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")
pc_grid = pc.grid(0.5)

interp = pc_grid.interpolate("z", "max")

plt.imshow(interp)
plt.colorbar()
plt.show()



