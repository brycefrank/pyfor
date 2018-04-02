import laspy
import numpy as np
import pandas as pd
import copy


# I want to read a las file into a numpy array
las = laspy.file.File("/home/bryce/Desktop/pyfor_test_data/PC_001.las")

points = np.stack((las.x, las.y, las.z, las.intensity, las.flag_byte, las.classification,
                   las.scan_angle_rank, las.user_data, las.pt_src_id), axis = 1)

# Get the laspy header object
header = las.header
print(points[:,0])
