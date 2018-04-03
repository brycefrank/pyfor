import pointcloud2
import laspy
import copy
from sys import getsizeof
import rasterizer


b = pointcloud2.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")

#print(b.grid(2).m)
#c = b.grid(2).get_missing_cells()
#print(c.loc[c['bins_x'] == 1])

c = b.grid(1).get_missing_cells()


b.plot(1)
