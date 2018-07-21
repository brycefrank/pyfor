import pyfor
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


test_cloud = pyfor.cloud.Cloud("/home/bryce/Programming/pyfor/pyfortest/data/test.las")
test_cloud.normalize(cell_size=0.5)


def bin_layers(cloud):
    layer_bins = np.searchsorted(np.arange(0.5, cloud.las.max[2] + 1), cloud.las.points['z'])
    return(layer_bins)


def get_layer(points, layer_index):
    return points.loc[points['bins_z'] == layer_index]


test_cloud.las.points['bins_z'] = bin_layers(test_cloud)


# Clustering algorthims were then applied to lowest 3 layers, any point without -1 index is considered vegetation and
# Removed
def get_non_veg_indices(layer_index, points):
    layer_xy = points.loc[points['bins_z'] == layer_index]
    db = DBSCAN(eps=0.3, min_samples=10).fit(layer_xy)
    non_veg_inds = layer_xy.index.values[np.where(db.labels_ == -1)]
    return(non_veg_inds)

def remove_veg(points, veg_layers = (0, 1, 2)):
    """

    :param points:
    :param veg_layers: A list of vegetation layers to modify. Ayrey uses the first 3 by default.
    :return:
    """

    non_veg_indices = [get_non_veg_indices(veg_layer, points) for veg_layer in veg_layers]
    non_veg_indices = np.concatenate(non_veg_indices).ravel()
    other_layer_indices = points.index.values[np.where(points['bins_z'] > veg_layers[-1])]
    keep_indices = np.concatenate([non_veg_indices, other_layer_indices])

    return(keep_indices)


# A canopy height model with a resolution of 1m was then developed over the study areas.
# This was smoothed with a 3x3 cell window and local maxima were detected using
# a 3m fixed radius window

test_cloud.las.points = test_cloud.las.points.iloc[remove_veg(test_cloud.las.points)]
chm = test_cloud.chm(1, interp_method= "nearest", pit_filter= "median")

# The local maxima are used as seed points, we should reproject these back to cloud space
seed_points = np.where(chm.local_maxima() != 0)
seed_xy = seed_points[0] + chm._affine[2], seed_points[1] + chm._affine[5]
seed_xy = np.stack(seed_xy, axis = 1)


# Use seed_xy as seed points for vlusters

test_layer = get_layer(test_cloud.las.points, 0)
KMeans(n_clusters= 3,init = seed_xy).fit(test_layer[['x', 'y']])