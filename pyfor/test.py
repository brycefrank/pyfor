import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)



pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")
pc_grid = pc.grid(1)
A = pc_grid.interpolate("min", "z")[100:200, 100:200]
pyfor.filter.zhang(A, 20)
np.sum(flag)