# An update of the cloudinfo class

import laspy
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import ogr
from pyfor import rasterizer
from pyfor import clip_funcs
from pyfor import plot

class CloudData:
    """
    A simple class composed of a numpy array of points and a laspy header, meant for internal use. This is basically
    a way to load data from the las file into memory.
    """
    def __init__(self, points, header):
        self.points = points
        self.x = self.points["x"]
        self.y = self.points["y"]
        self.z = self.points["z"]

        self.header = header

        self.min = [np.min(self.x), np.min(self.y), np.min(self.z)]
        self.max = [np.max(self.x), np.max(self.y), np.max(self.z)]
        self.count = np.alen(self.points)

    def write(self, path):
        """
        Writes the points and header to a .las file.

        :param path: The path of the .las file to write to.
        """
        # Make header manager
        writer = laspy.file.File(path, header = self.header, mode = "w")
        writer.x = self.points["x"]
        writer.y = self.points["y"]
        writer.z = self.points["z"]
        writer.intesity = self.points["intensity"]
        writer.classification = self.points["classification"]
        writer.flag_byte = self.points["flag_byte"]
        writer.scan_angle_rank = self.points["scan_angle_rank"]
        writer.user_data = self.points["user_data"]
        writer.pt_src_id = self.points["pt_src_id"]
        writer.close()

    def _update(self):
        self.min = [np.min(self.x), np.min(self.y), np.min(self.z)]
        self.max = [np.max(self.x), np.max(self.y), np.max(self.z)]
        self.count = np.alen(self.points)

class Cloud:
    def __init__(self, las):
        """
        A dataframe representation of a point cloud, with some useful functions for manipulating and displaying.

        :param las: A path to a las file, a laspy.file.File object, or a CloudFrame object
        """
        if type(las) == str:
            las = laspy.file.File(las)
            # Rip points from laspy
            points = pd.DataFrame({"x": las.x, "y": las.y, "z": las.z, "intensity": las.intensity, "classification": las.classification,
                                   "flag_byte":las.flag_byte, "scan_angle_rank":las.scan_angle_rank, "user_data": las.user_data,
                                   "pt_src_id": las.pt_src_id})
            header = las.header

            # Toss out the laspy object in favor of CloudData

            # FIXME the following line produces errors when creating new CloudData objects.
            # TODO one way to do this might be to move the above code into the __init__ for CloudData and handle close
            # there
            # las.close()
            self.las = CloudData(points, header)
        elif type(las) == CloudData:
            self.las = las
        else:
            print("Object type not supported, please input either a las file path or a CloudData object.")

        # We're not sure if this is true or false yet
        self.normalized = None
        self.crs = None

    def grid(self, cell_size):
        """
        Generates a Grid object for this Cloud given a cell size. See the documentation for Grid for more information.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :return: A Grid object.
        """
        return(rasterizer.Grid(self, cell_size))

    def plot(self, cell_size = 1, cmap = "viridis", return_plot = False):
        """
        Plots a basic canopy height model of the Cloud object. This is mainly a convenience function for \
        rasterizer.Grid.plot, check that method docstring for more information and more robust usage cases.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object.
        """
        if return_plot == True:
            return(rasterizer.Grid(self, cell_size).plot("max", return_plot= True))

        rasterizer.Grid(self, cell_size).plot("max", cmap, dim = "z")

    def iplot3d(self, max_points=30000, point_size=0.5, dim="z", colorscale="Virids"):
        """
        Plots the 3d point cloud in a compatible version for Jupyter notebooks using Plotly as a backend. If \
        max_points exceeds 30,000, the point cloud is downsampled using a uniform random distribution by default. \
        This can be changed using the max_points argument.

        :param max_points: The maximum number of points to render.
        :param point_size: The point size of the rendered point cloud.
        """
        self.min = [np.min(self.las.x), np.min(self.las.y), np.min(self.las.z)]
        self.max = [np.max(self.las.x), np.max(self.las.y), np.max(self.las.z)]
        self.count = np.alen(self.las.points)
        plot.iplot3d(self.las, max_points, point_size, dim, colorscale)

    def plot3d(self, point_size=1, cmap='Spectral_r', max_points=5e5):
        """
        Plots the three dimensional point cloud using a method suitable for non-Jupyter use (i.e. via the Python \
        console). By default, if the point cloud exceeds 5e5 points, then it is downsampled using a uniform random \
        distribution of 5e5 points. This is for performance purposes.

        :param point_size: The size of the rendered points.
        :param cmap: The matplotlib color map used to color the height distribution.
        :param max_points: The maximum number of points to render.
        """

        # Randomly sample down if too large
        if self.las.count > max_points:
                sample_mask = np.random.randint(self.las.header.count,
                                                size = int(max_points))
                #TODO update this to new pandas framework
                coordinates = np.stack([self.las.x, self.las.y, self.las.z], axis = 1)[sample_mask,:]
                print("Too many points, down sampling for 3d plot performance.")
        else:
            # TODO update this to new pandas framework
            coordinates = np.stack([self.las.x, self.las.y, self.las.z], axis = 1)

        # Start Qt app and widget
        pg.mkQApp()
        view = gl.GLViewWidget()

        # Normalize Z to 0-1 space
        z = np.copy(coordinates[:,2])
        z = (z - min(z)) / (max(z) - min(z))

        # Get matplotlib color maps
        cmap = cm.get_cmap(cmap)
        colors = cmap(z)

        # Create the points, change to opaque, set size to 1
        points = gl.GLScatterPlotItem(pos = coordinates, color = colors)
        points.setGLOptions('opaque')
        points.setData(size = np.repeat(point_size, len(coordinates)))

        # Add points to the viewer
        view.addItem(points)

        # Center on the aritgmetic mean of the point cloud and display
        # TODO Calculate an adequate zoom out distance
        center = np.mean(coordinates, axis = 0)
        view.opts['center'] = pg.Vector(center[0], center[1], center[2])
        view.show()

    def normalize(self, cell_size, num_windows = 7, dh_max = 2.5, dh_0 = 1, interp_method = "nearest"):
        """
        Normalizes this cloud object **in place** by generating a DEM using the default filtering algorithm  and \
        subtracting the underlying ground elevation. This uses a grid-based progressive morphological filter developed \
        in Zhang et al. (2003).

        This algorithm is actually implemented on a raster of the minimum Z value in each cell, but is included in \
        the Cloud object as a convenience wrapper. Its implementation involves creating a bare earth model and then \
        subtracting the underlying ground from each point's elevation value.

        If you would like to create a bare earth model, look instead toward Grid.ground_filter.

        Note that this current implementation is best suited for larger tiles. Best practices suggest creating a BEM \
        at the largest scale possible first, and using that to normalize plot-level point clouds in a production \
        setting.

        :param cell_size: The cell_size at which to rasterize the point cloud into bins, in the same units as the \
        input point cloud.
        :param num_windows: The number of windows to consider.
        :param dh_max: The maximum height threshold.
        :param dh_0: The null height threshold.
        :param interp_method: The interpolation method used to fill in missing values after the ground filtering \
        takes place. One of any: "nearest", "linear", or "cubic".
        """
        grid = self.grid(cell_size)
        dem_grid = grid.normalize(num_windows, dh_max, dh_0, interp_method)

        self.las.points['z'] = dem_grid.data['z']
        self.las.min = [np.min(dem_grid.data.x), np.min(dem_grid.data.y), np.min(dem_grid.data.z)]
        self.las.max = [np.max(dem_grid.data.x), np.max(dem_grid.data.y), np.max(dem_grid.data.z)]
        self.normalized = True

    def clip(self, geometry):
        """
        Clips the point cloud to the provided geometry (see below for compatible types) using a ray casting algorithm.

        :param geometry: Either a tuple of bounding box coordinates (square clip), an OGR geometry (polygon clip), \
        or a tuple of a point and radius (circle clip).
        :return: A new Cloud object clipped to the provided geometry.
        """
        if type(geometry) == tuple and len(geometry) == 4:
            # Square clip
            mask = clip_funcs.square_clip(self, geometry)
            keep_points = self.las.points.iloc[mask]

        elif type(geometry) == ogr.Geometry:
            keep_points = clip_funcs.poly_clip(self, geometry)

        return Cloud(CloudData(keep_points, self.las.header))

    def filter(self, min, max, dim):
        """
        Filters a cloud object for a given dimension **in place**.

        :param min: Minimum dimension to retain.
        :param max: Maximum dimension to retain.
        :param dim: The dimension of interest as a string. For example "z". This corresponds to a column label in \
        self.las.points dataframe.
        """
        condition = (self.las.points[dim] > min) & (self.las.points[dim] < max)
        self.las = CloudData(self.las.points[condition], self.las.header)

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

