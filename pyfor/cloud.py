# An update of the cloudinfo class

import laspy
import plyfile
import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from pyfor import rasterizer
from pyfor import clip_funcs
from pyfor import plot
import pathlib

# General class
class CloudData:
    def __init__(self, points, header):
        self.header = header
        self.points = points
        self.x = self.points["x"]
        self.y = self.points["y"]
        self.z = self.points["z"]
        self.min = [np.min(self.x), np.min(self.y), np.min(self.z)]
        self.max = [np.max(self.x), np.max(self.y), np.max(self.z)]
        self.count = np.alen(self.points)

    def _update(self):
        self.min = [np.min(self.x), np.min(self.y), np.min(self.z)]
        self.max = [np.max(self.x), np.max(self.y), np.max(self.z)]
        self.count = np.alen(self.points)

    def _append(self, other):
        """
        Append one CloudData object to another.
        :return:
        """
        self.points = pd.concat([self.points, other.points])
        self._update()

class PLYData(CloudData):
    def write(self, path):
        #coordinate_array = self.points[["x", "y", "z"]].values.T
        #vertex_array = list(zip(coordinate_array[0],coordinate_array[1], coordinate_array[2]))
        #vertex_array = np.array(vertex_array, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_array = self.points.to_records(index=False)
        elements = plyfile.PlyElement.describe(vertex_array, 'vertex')
        plyfile.PlyData([elements]).write(path)

class LASData(CloudData):
    def write(self, path):
        writer = laspy.file.File(path, header = self.header, mode = "w")
        writer.x = self.points["x"]
        writer.y = self.points["y"]
        writer.z = self.points["z"]
        writer.return_num = self.points["return_num"]
        writer.intesity = self.points["intensity"]
        writer.classification = self.points["classification"]
        writer.flag_byte = self.points["flag_byte"]
        writer.scan_angle_rank = self.points["scan_angle_rank"]
        writer.user_data = self.points["user_data"]
        writer.pt_src_id = self.points["pt_src_id"]
        writer.close()

class Cloud:
    """
    The cloud object is the integral unit of pyfor, and is where most of the action takes place. Many of the following \
    attributes are convenience functions for other classes and modules.
    """
    def __init__(self, path):

        if type(path) == str or type(path) == pathlib.PosixPath:
            self.filepath = path
            self.extension = os.path.splitext(path)[1]

            if self.extension == '.las':
                las = laspy.file.File(path)
                points = pd.DataFrame({"x": las.x, "y": las.y, "z": las.z, "intensity": las.intensity, "return_num": las.return_num, "classification": las.classification,
                                       "flag_byte":las.flag_byte, "scan_angle_rank":las.scan_angle_rank, "user_data": las.user_data,
                                       "pt_src_id": las.pt_src_id})
                header = las.header
                self.data = LASData(points, header)


            elif self.extension == '.ply':
                ply = plyfile.PlyData.read(path)
                ply_points = ply.elements[0].data
                points = pd.DataFrame({"x": ply_points["x"], "y": ply_points["y"], "z": ply_points["z"]})

                try:
                    points['red'] = ply_points['red']
                    points['green'] = ply_points['green']
                    points['blue'] = ply_points['blue']
                except ValueError:
                    pass

                header = None
                self.data = PLYData(points, header)

        elif type(path) == CloudData:
            self.data = path
        else:
            print("Object type not supported, please input either a las file path or a CloudData object.")

        # We're not sure if this is true or false yet
        self.normalized = None
        self.crs = None

    def __str__(self):
        """
        Returns a human readable summary of the Cloud object.
        """
        from os.path import getsize

        # Format max and min
        min =  [float('{0:.2f}'.format(elem)) for elem in self.data.min]
        max =  [float('{0:.2f}'.format(elem)) for elem in self.data.max]
        filesize = getsize(self.filepath)

        # TODO: Incorporate this in CloudData somehow, messy!
        if self.extension == '.las':
            las_version = self.data.header.version
            out = """ File Path: {}\nFile Size: {}\nNumber of Points: {}\nMinimum (x y z): {}\nMaximum (x y z): {}\nLas Version: {}
            
            """.format(self.filepath, filesize, self.data.count, min, max, las_version)
        elif self.extension == '.ply':
            out = """ File Path: {}\nFile Size: {}\nNumber of Points: {}\nMinimum (x y z): {}\nMaximum (x y z): {}""".format(self.filepath, filesize, self.data.count, min, max)

        return(out)

    def _discrete_cmap(self, n_bin, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, n_bin))
        cmap_name = base.name + str(n_bin)
        return LinearSegmentedColormap.from_list(cmap_name, color_list, n_bin)

    def _set_discrete_color(self, n_bin, series):
        """Adds a column 'random_id' to Cloud.las.points that reduces the 'user_data' column to a fewer number of random
        integers. Used to produce clearer 3d visualizations of detected trees.

        :param n_bin: Number of bins to reduce to.
        :param series: The pandas series to reduce, usually 'user_data' which is set to a unique tree ID after detection.
        """

        random_ints = np.random.randint(1, n_bin + 1, size = len(np.unique(series)))
        pre_merge = pd.DataFrame({'unique_id': series.unique(), 'random_id': random_ints})


        self.data.points = pd.merge(self.data.points, pre_merge, left_on = 'user_data', right_on = 'unique_id')

    def grid(self, cell_size):
        """
        Generates a Grid object for this Cloud given a cell size. The Grid is generally used to compute Raster objects
        See the documentation for Grid for more information.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :return: A Grid object.
        """
        return(rasterizer.Grid(self, cell_size))

    def plot(self, cell_size = 1, cmap = "viridis", return_plot = False, block=False):
        """
        Plots a basic canopy height model of the Cloud object. This is mainly a convenience function for \
        rasterizer.Grid.plot, check that method docstring for more information and more robust usage cases (i.e. \
        pit filtering and interpolation methods).

        :param cellf vmin or vmax is not given, they are initialized from the minimum and maximum value respectively of the first input processed. That is, __call__(A) calls autoscale_None(A). If clip _size: The resolution of the plot in the same units as the input file.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object. Not yet implemented.
        """

        rasterizer.Grid(self, cell_size).raster("max", "z").plot(cmap, block = block, return_plot = return_plot)

    def iplot3d(self, max_points=30000, point_size=0.5, dim="z", colorscale="Viridis"):
        """
        Plots the 3d point cloud in a compatible version for Jupyter notebooks using Plotly as a backend. If \
        max_points exceeds 30,000, the point cloud is downsampled using a uniform random distribution by default. \
        This can be changed using the max_points argument.

        :param max_points: The maximum number of points to render.
        :param point_size: The point size of the rendered point cloud.
        :param dim: The dimension on which to color (i.e. "z", "intensity", etc.)
        :param colorscale: The Plotly colorscale with which to color.
        """
        plot.iplot3d(self.data, max_points, point_size, dim, colorscale)

    def plot3d(self, dim = "z", point_size=1, cmap='Spectral_r', max_points=5e5, n_bin=8, plot_trees=False):
        """
        Plots the three dimensional point cloud using a method suitable for non-Jupyter use (i.e. via the Python \
        console). By default, if the point cloud exceeds 5e5 points, then it is downsampled using a uniform random \
        distribution of 5e5 points. This is for performance purposes.

        :param point_size: The size of the rendered points.
        :param dim: The dimension upon which to color (i.e. "z", "intensity", etc.)
        :param cmap: The matplotlib color map used to color the height distribution.
        :param max_points: The maximum number of points to render.
        """

        from pyqtgraph.Qt import QtCore, QtGui
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl

        # Randomly sample down if too large
        if dim == 'user_data' and plot_trees:
            dim = 'random_id'
            self._set_discrete_color(n_bin, self.data.points['user_data'])
            cmap = self._discrete_cmap(n_bin, base_cmap=cmap)

        if self.data.count > max_points:
                sample_mask = np.random.randint(self.data.count,
                                                size = int(max_points))
                coordinates = np.stack([self.data.points.x, self.data.points.y, self.data.points.z], axis = 1)[sample_mask,:]

                color_dim = np.copy(self.data.points[dim].iloc[sample_mask].values)
                print("Too many points, down sampling for 3d plot performance.")
        else:
            coordinates = np.stack([self.data.points.x, self.data.points.y, self.data.points.z], axis = 1)
            color_dim = np.copy(self.data.points[dim].values)

        # If dim is user data (probably TREE ID or some such thing) then we want a discrete colormap
        if dim != 'random_id':
            color_dim = (color_dim - np.min(color_dim)) / (np.max(color_dim) - np.min(color_dim))
            cmap = cm.get_cmap(cmap)
            colors = cmap(color_dim)

        else:
            colors = cmap(color_dim)

        # Start Qt app and widget
        pg.mkQApp()
        view = gl.GLViewWidget()


        # Create the points, change to opaque, set size to 1
        points = gl.GLScatterPlotItem(pos = coordinates, color = colors)
        points.setGLOptions('opaque')
        points.setData(size = np.repeat(point_size, len(coordinates)))

        # Add points to the viewer
        view.addItem(points)

        # Center on the aritgmetic mean of the point cloud and display
        center = np.mean(coordinates, axis = 0)
        view.opts['center'] = pg.Vector(center[0], center[1], center[2])
        # Very ad-hoc
        view.opts['distance'] = (self.data.max[0] - self.data.min[0]) * 1.2
        #return(view.opts)
        view.show()

    def normalize(self, cell_size, **kwargs):
        """
        Normalizes this cloud object **in place** by generating a DEM using the default filtering algorithm  and \
        subtracting the underlying ground elevation. This uses Kraus and Pfeifer (1998). This is a convenience \
        wrapper for `ground_filter.KrausPfeifer1998.normalize`. See that documentation for more information.

        :param cell_size: The cell_size at which to rasterize the point cloud into bins, in the same units as the \
        input point cloud.
        :param kwargs: Keyword arguments to `ground_filter.KrausPfeifer1998`
        """
        from pyfor.ground_filter import KrausPfeifer1998
        filter = KrausPfeifer1998(self, cell_size, **kwargs)
        filter.normalize(cell_size)

    def clip(self, poly):
        """
        Clips the point cloud to the provided shapely polygon using a ray casting algorithm.

        :param poly: A shapely polygon in the same CRS as the Cloud.
        :return: A new cloud object clipped to the provided polygon.
        """
        #TODO Implement geopandas for multiple clipping polygons.

        keep = clip_funcs.poly_clip(self.data.points, poly)
        # Create copy to avoid warnings
        keep_points = self.data.points.iloc[keep].copy()

        new_cloud =  Cloud(CloudData(keep_points, self.data.header))
        new_cloud.data._update()

        return new_cloud

    def filter(self, min, max, dim):
        """
        Filters a cloud object for a given dimension **in place**.

        :param min: Minimum dimension to retain.
        :param max: Maximum dimension to retain.
        :param dim: The dimension of interest as a string. For example "z". This corresponds to a column label in \
        self.las.points dataframe.
        """
        condition = (self.data.points[dim] > min) & (self.data.points[dim] < max)

        self.data = CloudData(self.data.points[condition], self.data.header)
        self.data._update()

    def chm(self, cell_size, interp_method=None, pit_filter=None, kernel_size=3):
        """
        Returns a Raster object of the maximum z value in each cell.

        :param cell_size: The cell size for the returned raster in the same units as the parent Cloud or las file.
        :param interp_method: The interpolation method to fill in NA values of the produced canopy height model, one \
        of either "nearest", "cubic", or "linear"
        :param pit_filter: If "median" passes a median filter over the produced canopy height model.
        :param kernel_size: The kernel size of the median filter, must be an odd integer.
        :return: A Raster object of the canopy height model.
        """

        if pit_filter == "median":
            raster = self.grid(cell_size).interpolate("max", "z", interp_method=interp_method)
            raster.pit_filter(kernel_size=kernel_size)
            return raster

        if interp_method==None:
            return(self.grid(cell_size).raster("max", "z"))

        else:
            return(self.grid(cell_size).interpolate("max", "z", interp_method))

    @property
    def convex_hull(self):
        """
        Calculates the convex hull of the 2d plane.

        :return: A single-element geoseries of the convex hull.
        """
        from scipy.spatial import ConvexHull
        import geopandas as gpd
        from shapely.geometry import Polygon

        hull = ConvexHull(self.data.points[["x", "y"]].values)
        hull_poly = Polygon(hull.points[hull.vertices])

        return gpd.GeoSeries(hull_poly)

    def write(self, path):
        """
        Write the Cloud to a las file.

        :param path: The path of the output file.
        :return:
        """
        self.data.write(path)

