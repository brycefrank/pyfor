import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)

pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")

pc_grid = pc.grid(0.5)

dem = pc_grid.ground_filter(7, 2.5, 0.5)

new = pc_grid.normalize(7, 2.5, 0.5)
new.plot("max")

pc_grid = pc.grid(1)
