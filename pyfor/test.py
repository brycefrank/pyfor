import laspy
import copy
from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
import itertools

pc = cloud.Cloud("/home/bryce/Desktop/pyfor_test_data/plot_tiles/PC110201LeafOn2010.LAS")

point_array = pc.las.points

mask = point_array[:,2] > 300

print(point_array[mask].shape)