import pyfor
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt
from shapely.geometry import asMultiPoint
import geopandas as gpd

class LayerStacking:
    # TODO Make inherent from a generic tree detection class
    """
    An implementation of Ayrey et al. (2017) layer stacking algorithm.
    """
    def __init__(self, cloud, chm_resolution = 1, n_jobs = 1, buffer_distance = 0.5):
        """

        :param cloud:
        :param chm_resolution:
        :param n_jobs: The number of threads to run in paralell.
        """
        self.cloud = cloud
        self.points = self.cloud.las.points
        self.chm = self.cloud.chm(chm_resolution, interp_method= "nearest", pit_filter= "median")
        self.n_jobs = n_jobs
        self.buffer_distance = buffer_distance

        # TODO Make into functions below?
        # FIXME this should be generalized for unit (i.e. point clouds in feet or in meters)
        # Bin the layers in the cloud
        layer_bins = np.searchsorted(np.arange(0.5, self.cloud.las.max[2] + 1), self.points['z'])
        self.points['bins_z'] = layer_bins
        self.n_layers = len(np.unique(layer_bins))

    def project_indices(self, indices):
        """
        Converts indices of an array (for example, those indices that describe the location of a local maxima) to the
        same space as the input cloud object. Assumes the array has already been flipped upside down.

        :param indices: The indices to project, an Nx2 matrix of indices where the first column are the rows (Y) and
        the second column is the columns (X)
        :param raster: The parent raster object
        :return:
        """

        seed_xy = indices[:,1] + (self.chm._affine[2] / self.chm._affine[0]), \
                  indices[:,0] + (self.chm._affine[5] - (self.chm.grid.las.max[1] - self.chm.grid.las.min[1]) / abs(self.chm._affine[4]))
        seed_xy = np.stack(seed_xy, axis = 1)
        return(seed_xy)

    def get_detected_top_coordinates(self, min_distance = 3, threshold_abs=3):
        """
        Gets the coordinates of detected tops.
        :return:
        """

        top_indices = corner_peaks(np.flipud(self.chm.local_maxima(min_distance = min_distance,
                                                                   threshold_abs=threshold_abs) !=0))
        self.top_coordinates = self.project_indices(top_indices)

    def get_layer(self, layer_index):
        """
        Returns a given layer.

        :param layer_index: The layer which to return (starting from 0).
        :return:
        """
        return self.points.loc[self.points['bins_z'] == layer_index]

    def get_non_veg_indices(self, layer_index):
        """
        Retrieves the non-vegetation indices. These are the points that are kept for further analysis. Used as a
        subroutine for self.remove_veg (see below).

        :param points:
        :return:
        """

        layer_xy = self.points.loc[self.points['bins_z'] == layer_index]
        db = DBSCAN(eps=0.3, min_samples=10).fit(layer_xy)
        non_veg_inds = layer_xy.index.values[np.where(db.labels_ == -1)]
        return(non_veg_inds)

    def remove_veg(self, veg_layers = (0, 1, 2)):
        """
        Removes vegetation from the input cloud object.

        :param veg_layers: An iterable of the layers to consider as vegetation (Ayrey recommends the first 3), starting
        at 0.
        :return: The indices to keep.
        """

        non_veg_indices = [self.get_non_veg_indices(veg_layer) for veg_layer in veg_layers]
        non_veg_indices = np.concatenate(non_veg_indices).ravel()
        other_layer_indices = self.points.index.values[np.where(self.points['bins_z'] > veg_layers[-1])]
        keep_indices = np.concatenate([non_veg_indices, other_layer_indices])

        return(keep_indices)

    def cluster_layer(self, layer_index):
        """
        Performs k-means clustering on an input layer
        :param layer_index:
        :return:
        """
        print("Clustering layer {}".format(layer_index + 1))
        layer = self.get_layer(layer_index)
        if len(layer) >= self.top_coordinates.shape[0]:
            clusters = KMeans(n_clusters=self.top_coordinates.shape[0], init = self.top_coordinates, n_jobs=self.n_jobs,
                              ).fit(layer[['x', 'y']])
            return(clusters)
        else:
            return(None)

    def cluster_all_layers(self):
        """
        Clusters every layer present in the parent cloud object.

        :return: A list of cluster objects, one for each layer in the cloud object.
        """
        # TODO handle the case where the number of points is less than the number of tree otops
        # FIXME seems to be calling cluster layer twice!
        return [self.cluster_layer(i) for i in range(self.n_layers) if self.cluster_layer(i) is not None]

    def buffer_cluster_layers(self):
        """
        Converts clusters to points and buffers
        :return:
        """
        clustered_layers = self.cluster_all_layers()
        #clustered_layers[0].labels_

        this = asMultiPoint(self.get_layer(0)[["x", "y"]].values)

        # Buffer the points and assign clusters
        # TODO iterate
        # TODO do labels even matter? It seems we are just tryingto produce the overlap map
        # TODO fill holes
        buffer_points = gpd.GeoSeries(this.geoms).buffer(self.buffer_distance)
        clustered_geoms = gpd.GeoDataFrame({'cluster_label': clustered_layers[0].labels_, 'geometry': buffer_points})
        clustered_geoms = clustered_geoms.dissolve(by = "cluster_label")

        return([clustered_geoms])

    def overlap_polygons(self):
        # Get list of polygon layers
        layers_of_polygons = self.buffer_cluster_layers()

        # Rasterize each.
        pass

    def detect(self):
        """
        Executes the detection algoritm on the input point cloud with set parameters.
        :return:
        """
        pass
