import importlib.machinery
import numpy as np
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()



pc = pyfor.cloud.Cloud("/home/bryce/Programming/PyFor/samples/data/NEON_D03_OSBS_DP1_405000_3276000_classified_point_cloud.laz")

pc_grid = pc.grid(1)

pc_grid.metrics(my_pct, "z")

pc_grid.interpolate(my_pct, "z")

def my_pct(a):
    return(np.percentile(a, q = 0.2))