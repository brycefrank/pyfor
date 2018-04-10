import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)

pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")
pc_grid = pc.grid(0.5)

# 827244 cells with values
pc_grid.cells.agg({"z": "count"})

# 858000 cells total
# 28899 cells missing

pc_grid.empty_cells


A = pc_grid.interpolate("min", "z")
pyfor.filter.zhang(A, 5, 3, 0.5, 0.5, pc_grid)

