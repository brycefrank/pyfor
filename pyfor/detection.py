import pyfor
import numpy as np

# Implementing dalponte2016-1
test_cloud = pyfor.cloud.Cloud("/home/bryce/Programming/pyfor/pyfortest/data/test.las")
test_cloud.normalize(cell_size=0.5)

test_cloud.plot3d()

lpf_chm = test_cloud.chm(0.5, interp_method="nearest", pit_filter="median", kernel_size=5)



from skimage.morphology import watershed


watershed_array = np.flipud(lpf_chm.array)
tops = lpf_chm.local_maxima(min_distance=2, threshold_abs=2)
L = watershed(-watershed_array, tops, mask=watershed_array)

# from each region in L the first return ALS points are extracted
# First, classify the point cloud

xy = lpf_chm.grid.data[["bins_x", "bins_y"]].values
tree_id = L[xy[:, 1], xy[:, 0]]

# Update the CloudData and Grid objects
lpf_chm.grid.las.points["user_data"] = tree_id
lpf_chm.grid.data = lpf_chm.grid.las.points
lpf_chm.grid.cells = lpf_chm.grid.data.groupby(['bins_x', 'bins_y'])

lpf_chm.grid.las.points = lpf_chm.grid.las.points.loc[lpf_chm.grid.las.points['return_num'] == 1]

# Iterate through each tree clump and apply Otsu threshold
from skimage import filters

lpf_chm.grid.cloud.chm(0.5, interp_method="nearest", pit_filter="median", kernel_size=5).array

filters.threshold_otsu()


points = lpf_chm.grid.las.points
points.loc[points['bins_x'] == np.where(1 == L)[0]]

