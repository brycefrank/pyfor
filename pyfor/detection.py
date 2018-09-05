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
                 weights = (2, 3, 4, 4), overlap_kernel_size = 3, scnd_pass_min_dist = 3, scnd_pass_threshold_abs = 3,
                 remove_veg=True):
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
        self.first_pass_threshold_abs = first_pass_threshold_abs
        self.veg_layers = veg_layers
        self.percentiles = percentiles
        self.weights = weights
        self.overlap_kernel_size = overlap_kernel_size
        self.scnd_pass_min_dist = scnd_pass_min_dist
        self.scnd_pass_threshold_abs = scnd_pass_threshold_abs
        self.remove_veg = remove_veg

        # Bin the layers in the cloud from 0.5 to the maximum height
        layer_bins = np.searchsorted(np.arange(0.5, self.cloud.las.max[2] + 1), self.points['z'])
        self.points['bins_z'] = layer_bins
        self.n_layers = len(np.unique(layer_bins))

    @property
    def _top_coordinates(self, min_distance = 3, threshold_abs=3):
        """
        Gets the coordinates of detected tops.

        :param min_distance: The minimum distance to be used in pyfor.raster.Raster.local_maxima
        :param threshold_abs: The absolute threshold to be used in pyfor.raster.Raster.local_maxima
        :return: An Nx2 array of coordinates such that the first column is a vector of Y positions and the second column
        is a vector of X positions.
        """

        tops_raster = self.chm.local_maxima(min_distance=min_distance, threshold_abs=threshold_abs)
        top_indices = np.stack(np.where(tops_raster.array > 0)).T
        projected_indices = pyfor.gisexport.project_indices(top_indices, tops_raster)
        return projected_indices


    @property
    def _complete_layers(self):
        """
        A list of the complete layers. A layer is considered complete if it has more points than the number of top
        coordinates. This is useful for processes downstream in the algorithm because the KMeans algorithm will fail
        if this condition does not hold for a given layer.

        :return: A list of indices of all complete layers in the cloud.
        """

        num_points = self.points.groupby("bins_z").size()
        bool_layers = num_points > self._top_coordinates.shape[0]
        return(bool_layers[bool_layers==True].index.values)

    def _get_layer(self, layer_index):
        """
        Returns a given layer.

        :param layer_index: The layer which to return (starting from 0).
        :return:
        """

        return self.points.loc[self.points['bins_z'] == layer_index]

    def _get_non_veg_indices(self, layer_index):
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

    def _remove_veg(self):
        """
        Removes vegetation points from the input cloud object.

        :param veg_layers: An iterable of the layers to consider as vegetation (Ayrey recommends the first 3), starting
        at 0.
        :return: The indices to keep.
        """

        non_veg_indices = [self._get_non_veg_indices(veg_layer) for veg_layer in self.veg_layers]
        non_veg_indices = np.concatenate(non_veg_indices).ravel()
        other_layer_indices = self.points.index.values[np.where(self.points['bins_z'] > self.veg_layers[-1])]
        keep_indices = np.concatenate([non_veg_indices, other_layer_indices])
        self.points = self.points.iloc[keep_indices,:]

    def _cluster_layer(self, layer_index):
        """
        Performs k-means clustering on an input layer

        :param layer_index:
        :return:
        """
        print("Clustering layer {}".format(layer_index + 1))
        layer = self._get_layer(layer_index)
        clusters = KMeans(n_clusters=self._top_coordinates.shape[0], init = self._top_coordinates, n_jobs=self.n_jobs,
                              ).fit(layer[['x', 'y']])
        return(clusters.labels_)

    def _cluster_all_layers(self):
        """
        Clusters every complete layer in the cloud object.

        :return: A list of cluster objects, one for each layer in the cloud object.
        """

        # Get absolute indices of complete layers
        return [self._cluster_layer(i) for i in self._complete_layers]

    def _buffer_cluster_layers(self):
        """
        Buffers all points not removed after _remove_veg (or all points of self.remove_veg is set to False)
        :return:
        """
        # Subset to only complete layers
        multi_points = self.points[self.points['bins_z'].isin(self._complete_layers)]
        multi_points = asMultiPoint(multi_points[['x', 'y']].values)
        buffer_points = gpd.GeoDataFrame(gpd.GeoSeries(multi_points.geoms).buffer(self.buffer_distance))
        ## TODO handle for veg removal
        buffer_points["bins_z"] = self.points["bins_z"]
        labels = [item for sublist in self._cluster_all_layers() for item in sublist]
        buffer_points["labels"] = labels

        # Fix some GPD quirks
        buffer_points = buffer_points.set_geometry(0)
        buffer_points.geometry.geom_type = buffer_points[0].geom_type
        return(buffer_points)

    def _layer_inds_between_pct(self, lb, ub):
        """
        Gets layer indices between upper and lower bound quantiles (inclusive lower, exclusive upper).
        :return:
        """

        true_inds = list(np.where((self._complete_layers >= lb) & (self._complete_layers < ub))[0])
        all_inds = np.array(range(len(self._complete_layers)))
        return(all_inds[true_inds])


    def _construct_layer_weights(self, percentiles = (70, 80, 90, 100), weights = (2, 3, 4, 4)):
        """
        Ayrey uses a weighting system to assign higher point values to different layers. The top 70th percentile
        clusters receive double weight, the top 80th receive triple and the top 90th receive quadruple. This function
        returns a dictionary where each key is the index of the complete_layer and each value is its respective weight.
        This is used later to pass the value (i.e. the weight) of each layer to self.rasterize

        :param percentiles: A tuple of percentiles to consider for weighting.
        :param weights: A tuple of weights, must be the same length as `percentiles`
        :return: A dictionary of weights for each layer.
        """
        percentile_breaks = [np.percentile(self._complete_layers, percentile) for percentile in percentiles]
        n_complete = len(self._complete_layers)

        # For each break point and its neighbor, retrieve the complete layer indices
        weighted_layers = []
        for break_point_ind in range(len(percentile_breaks) - 1):
            weighted_layers.append(self._layer_inds_between_pct(percentile_breaks[break_point_ind], percentile_breaks[break_point_ind + 1]))
        weighted_layers.append([n_complete - 1])

        # Construct dictionary with layer index as key and weight as value, initiate with all 1s as values
        weight_dict = dict(zip(range(n_complete), np.repeat(1, n_complete)))

        # Modify values of keys for layers in weighted_layers
        for i in range(len(weighted_layers)):
            for j in weighted_layers[i]:
                weight_dict[j] = weights[i]

        return(weight_dict)

    def _rasterize(self, geodataframe, value):
        """
        Converts buffered points into rasterized

        :param geodataframe:
        :param value:
        :return:
        """
        transform = self.chm._affine

        # TODO may be re-usable for other features. Consider moving to gisexport
        # FIXME check for cell sizes that are not 1
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
        layers_of_polygons = self._buffer_cluster_layers()
        weights_dict = self._construct_layer_weights()

        # Rasterize each.
        for key, value in weights_dict.items():
            self._rasterize(layers_of_polygons[key], value)

        layers_of_rasters = [self._rasterize(layers_of_polygons[key], value) for key, value in weights_dict.items()]

        array = np.sum(np.dstack(layers_of_rasters), axis = 2)
        # Flip
        array = np.flipud(array)
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
        if self.remove_veg is not None:
            self._remove_veg()
        raster = self.get_overlap_map(smoothed=True)
        return(raster)


