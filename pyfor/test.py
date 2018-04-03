import pointcloud2
import laspy
import copy
from sys import getsizeof
import rasterizer
import numpy as np
import matplotlib.pyplot as plt
import itertools
import rasterio

# Create DefaultGTiffProfile
prof = rasterio.profiles.DefaultGTiffProfile()

b = pointcloud2.Cloud("/home/bryce/Desktop/pyfor_test_data/plot_tiles/PC107701LeafOn2010.LAS")
c = b.grid(0.5)

out_array = c.array("max", "z")

# do the silly example first...
x = np.linspace(-4.0, 4.0, 240)
y = np.linspace(-3.0, 3.0, 180)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-2 * np.log(2) * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 1 ** 2)
Z2 = np.exp(-3 * np.log(2) * ((X + 0.5) ** 2 + (Y + 0.5) ** 2) / 2.5 ** 2)
Z = 10.0 * (Z2 - Z1)
Z = np.array([[1,2,3],[1,2,3]])

from rasterio.transform import from_origin
res = (x[-1] - x[0]) / 240.0
transform = from_origin(x[0] - res / 2, y[-1] + res / 2, res, res)

new_dataset = rasterio.open('/home/bryce/Desktop/new.tif', 'w', driver='GTiff',
                            height=Z.shape[0], width=Z.shape[1],
                            count=1, dtype=Z.dtype,
                            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ', transform=transform)
