import importlib.machinery
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
pyfor = importlib.machinery.SourceFileLoader('pyfor','/home/bryce/Programming/PyFor/pyfor/__init__.py').load_module()
reload(pyfor)

pc = pyfor.cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")
pc.normalize(0.5)

pc_grid = pc.grid(0.5)
pc_grid.plot("max", "count")

norm_grid = pc_grid.normalize()


import pandas as pd
arr = pc_grid.ground_filter()

df = pd.DataFrame(arr.array).stack().rename_axis(['bins_y', 'bins_x']).reset_index(name='val')

pc_grid.data = pd.merge(pc_grid.data, df)

pc_grid.data['z'] = pc_grid.data['z'] - pc_grid.data['val']

pc_grid.cells = pc_grid.data.groupby(['bins_x', 'bins_y'])

plt.matshow(pc_grid._interpolate("max", "z"))
plt.show()