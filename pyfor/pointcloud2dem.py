# Constructs a digital elevation model.

import pandas as pd
from scipy.interpolate import griddata
import numpy as np



# Example at  https://docs.scipy.org/doc/scipy-0.18.1/reference/
# generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

points = np.random.rand(1000, 2)
values = func(points[:,0], points[:,1])

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

# My Attempt

df1 = pd.read_pickle(r"C:\pyfor\testpickles\df2.pkl")

print(df1[['x', 'y']].values)

grid_x, grid_y = np.mgrid[453193:454192:1, 4735490:4736309:1]

grid_z3 = griddata(df1[['x', 'y']].values, df1['z'].values, (grid_x, grid_y), method='nearest')

import matplotlib.pyplot as plt

plt.imshow(grid_z3, extent = (453193,454192,4735490,4736309), origin='lower')

plt.show()