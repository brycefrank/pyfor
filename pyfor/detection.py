import pyfor
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt
from shapely.geometry import asMultiPoint
import geopandas as gpd
from rasterio.io import MemoryFile
from rasterio import features

class LayerStacking:
    # TODO Make inherent from a generic tree detection class
    """
    An implementation of Ayrey et al. (2017) layer stacking algorithm.
    """
    def __init__(self, cloud, n_jobs = 1, chm_resolution = 1, buffer_distance = 0.5, first_pass_min_dist = 3,
                 first_pass_threshold_abs = 3, veg_layers = (0, 1, 2), percentiles = (70, 80, 90, 100),
                 weights = (2, 3, 4, 4), overlap_kernel_size = 3, scnd_pass_min_dist = 3, scnd_pass_threshold_abs = 3):
        # TODO implement arbitrary layer bin widths
        # TODO check if cloud has been normalized
        """

        :param cloud: The cloud object to detect on.
        :param n_jobs: The number of threads to conduct KMeans.
        :param chm_resolution: The CHM resolution for the initial detection.
        :param buffer_distance: Distance to buffer points.
        :param first_pass_min_dist: The minimum distance for initial top detection.
        :param first_pass_threshold_abs: The absolute threshold for initial top detection.
        :param veg_layers: Layers to be considered vegetation
        :param percentiles: Percentiles to weight.
        :param weights: Weights given to layers within provided percentiles.
        :param overlap_kernel_size: The smoothing kernel size for the overlap map (in pixels).
        :param scnd_pass_min_dist: The minimum distance for the final top detection.
        :param scnd_pass_threshold_abs: The absolute threshold for initial top detection.
        """
        print("Warning: LayerStacking is still in development. Use at your own risk.")

        # Meta Information
        self.cloud = cloud
        self.points = self.cloud.las.points
        self.chm = self.cloud.chm(chm_resolution, interp_method= "nearest", pit_filter= "median")

        # Algorithm parameters
        self.n_jobs = n_jobs
        self.buffer_distance = buffer_distance
        self.first_pass_min_dist = first_pass_min_dist

        # Bin the layers in the cloud from 0.5 to the maximum height
        layer_bins = np.searchsorted(np.arange(0.5, self.cloud.las.max[2] + 1), self.points['z'])
        self.points['bins_z'] = layer_bins
        self.n_layers = len(np.unique(layer_bins))

    def project_indices(self, indices):
        """
        Converts indices of an array (for example, those indices that describe the location of a local maxima) to the
        same space as the input cloud object. Assumes the array has already been flipped upside down.

        :param indices: The indices to project, an Nx2 matrix of indices where the first column are the rows (Y) and
        the second column is the columns (X)
        :return:
        """

        seed_xy = indices[:,1] + (self.chm._affine[2] / self.chm._affine[0]), \
                  indices[:,0] + (self.chm._affine[5] - (self.chm.grid.las.max[1] - self.chm.grid.las.min[1]) /
                                  abs(self.chm._affine[4]))
        seed_xy = np.stack(seed_xy, axis = 1)
        return(seed_xy)

    @property
    def top_coordinates(self, min_distance = 3, threshold_abs=3):
        """
        Gets the coordinates of detected tops.
        :return:
        """

        top_indices = corner_peaks(np.flipud(self.chm.local_maxima(min_distance = min_distance,
                                                                   threshold_abs=threshold_abs) !=0))
        return(self.project_indices(top_indices))

    @property
    def complete_layers(self):
        """
        A list of the complete layers. A layer is considered complete if it has more points than the number of top
        coordinates. This is useful for processes downstream in the algorithm because the KMeans algorithm will fail
        if this condition does not hold for a given layer.

        :return: A list of indices of all complete layers in the cloud.
        """

        num_points = self.points.groupby("bins_z").size()
        bool_layers = num_points > self.top_coordinates.shape[0]
        return(bool_layers[bool_layers==True].index.values)

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
        # TODO fill holes
        # TODO implement cluster labels
        # TODO this can be cleaned up by implementing one big GDF with a layer column
        multi_points = [asMultiPoint(self.get_layer(i)[["x", "y"]].values) for i in range(self.n_layers)]
        buffer_points = [gpd.GeoSeries(multi_points[i].geoms).buffer(self.buffer_distance) for i in range(len(multi_points))]
        clustered_geoms = [gpd.GeoDataFrame(buffer_points[i]) \
                           for i in range(len(buffer_points))]
        clustered_geoms = [clustered_geoms[i].set_geometry(0) for i in range(len(clustered_geoms))]
        return(clustered_geoms)

    def layer_inds_between_pct(self, lb, ub):
        """
        Gets layer indices between upper and lower bound quantiles (inclusive lower, exclusive upper).
        :return:
        """

        true_inds = list(np.where((self.complete_layers >= lb) & (self.complete_layers < ub))[0])
        all_inds = np.array(range(len(self.complete_layers)))
        return(all_inds[true_inds])


    def layer_weights(self, percentiles = (70, 80, 90, 100), weights = (2, 3, 4, 4)):
        """
        Ayrey uses a weighting system to assign higher point values to different layers. The top 70th percentile
        clusters receive double weight, the top 80th receive triple and the top 90th receive quadruple. This function
        returns a dictionary where each key is the index of the complete_layer and each value is its respective weight.
        This is used later to pass the value (i.e. the weight) of each layer to self.rasterize

        :param percentiles:
        :param weights:
        :return:
        """
        percentile_breaks = [np.percentile(self.complete_layers, percentile) for percentile in percentiles]
        n_complete = len(self.complete_layers)

        # For each break point and its neighbor, retrieve the complete layer indices
        weighted_layers = []
        for break_point_ind in range(len(percentile_breaks) - 1):
            weighted_layers.append(self.layer_inds_between_pct(percentile_breaks[break_point_ind], percentile_breaks[break_point_ind + 1]))
        weighted_layers.append([n_complete - 1])

        # Construct dictionary with layer index as key and weight as value, initiate with all 1s as values
        weight_dict = dict(zip(range(n_complete), np.repeat(1, n_complete)))

        # Modify values of keys for layers in weighted_layers
        for i in range(len(weighted_layers)):
            for j in weighted_layers[i]:
                weight_dict[j] = weights[i]

        return(weight_dict)

    def rasterize(self, geodataframe, value):
        transform = self.chm._affine

        # TODO may be re-usable for other features. Consider moving to gisexport
        with MemoryFile() as memfile:
            with memfile.open(driver='GTiff',
                              width = self.chm.array.shape[1],
                              height = self.chm.array.shape[0],
                              count = self.chm.grid.cell_size,
                              dtype = np.uint8,
                              nodata=0,
                              transform=transform) as out:

                shapes = ((geom, value) for geom, value in zip(geodataframe[0], np.repeat(value, len(geodataframe))))
                burned = features.rasterize(shapes = shapes, fill = 0, out_shape = (self.chm.array.shape[0], self.chm.array.shape[1]),
                                            transform=transform)

                memfile.close()
                return(burned)


    def get_overlap_map(self, smoothed = True, kernel_size = 3):
        """
        Overlaps the rasters generated by the buffered polygons.

        :return: A 2D numpy array of weighted overlaps in each cell
        """
        # Get list of polygon layers
        layers_of_polygons = self.buffer_cluster_layers()
        weights_dict = self.layer_weights()

        # Rasterize each.
        for key, value in weights_dict.items():
            self.rasterize(layers_of_polygons[key], value)

        layers_of_rasters = [self.rasterize(layers_of_polygons[key], value) for key, value in weights_dict.items()]

        array = np.sum(np.dstack(layers_of_rasters), axis = 2)
        raster = pyfor.rasterizer.Raster(array, self.chm.grid)

        if smoothed:
            raster.pit_filter(kernel_size=kernel_size)
            return(raster)
        else:
            return(raster)


    def detect(self):
        """
        Executes the detection algorithm on the input point cloud with set parameters.
        :return:
        """
        # TODO expose kernel and min_distance options to user in init
        self.remove_veg()
        raster = self.get_overlap_map(smoothed=True)
        return(raster.local_maxima(min_distance=6))


