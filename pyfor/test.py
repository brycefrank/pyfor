import importlib.machinery
import numpy as np
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()



pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/plot_tiles/PC107701LeafOn2010.LAS")

pc_grid = pc.grid(1)

pc_grid.metrics(np.max, "z")