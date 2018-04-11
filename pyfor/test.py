import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)

pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/plot_tiles/PC107701LeafOn2010.LAS")

pc_grid = pc.grid(0.5)

pc_grid = pc.grid(1)
