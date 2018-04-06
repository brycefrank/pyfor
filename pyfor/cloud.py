# An update of the cloudinfo class

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import json
import ogr
from pyfor import rasterizer
from pyfor import clip_funcs


class CloudData:
    """
    A simple class composed of a numpy array of points and a laspy header, meant for internal use. This is basically
    a way to load data from the las file into memory.
    """
    def __init__(self, points, header):
        # TODO naming is a bit convoluted, but functioning
        # TODO expand
        self.points = points
        self.x = self.points["x"]
        self.y = self.points["y"]
        self.z = self.points["z"]

        if type(header) == laspy.header.HeaderManager:
            self.header = header.copy()
        else:
            self.header = header

        self.header.min = [np.min(self.x), np.min(self.y), np.min(self.z)]
        self.header.max = [np.max(self.x), np.max(self.y), np.max(self.z)]
        self.header.count = np.alen(self.points)


class Cloud:
    def __init__(self, las):
        """
        A dataframe representation of a point cloud.

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
            self.las = CloudData(points, header)
        elif type(las) == CloudData:
            self.las = las
        else:
            print("Object type not supported, please input either a las file path or a CloudData object.")

    def grid(self, cell_size):
        """
        Generates a Grid object for this Cloud given a cell size.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :return: A Grid object.
        """
        return(rasterizer.Grid(self, cell_size))

    def plot(self, cell_size = 1, return_plot = False):
        """
        Plots a basic canopy height model of the Cloud object. This is mainly a convenience function for
        rasterizer.Grid.plot, check that method docstring for more information and more robust usage cases.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object.
        """

        rasterizer.Grid(self, cell_size).plot("max")

        if return_plot == True:
            return(rasterizer.Grid(self, "max", cell_size, return_plot = True))

    def plot3d(self, point_size = 1, cmap = 'Spectral_r', max_points = 5e5):
        """
        Plots the three dimensional point cloud. By default, if the point cloud exceeds 5e5 points, then it is
        downsampled using a uniform random distribution of 5e5 points.

        :param point_size: The size of the rendered points.
        :param cmap: The matplotlib color map used to color the height distribution.
        :param max_points: The maximum number of points to render.
        """

        # Randomly sample down if too large
        if self.las.header.count > max_points:
                sample_mask = np.random.randint(self.las.header.count,
                                                size = int(max_points))
                coordinates = np.stack([self.las.x, self.las.y, self.las.z], axis = 1)[sample_mask,:]
                print("Too many points, down sampling for 3d plot performance.")
        else:
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

    def clip(self, geometry):
        """
        Returns a new Cloud object clipped to the provided geometry
        :param geometry: Either a tuple of bounding box coordinates (square clip), an OGR geometry (polygon clip),
        or a tuple of a point and radius (circle clip)
        :return:
        """
        # TODO Could be nice to add a warning if the shapefile extends beyond the pointcloud bounds

        if type(geometry) == tuple and len(geometry) == 4:
            # Square clip
            mask = clip_funcs.square_clip(self, geometry)
            keep_points = self.las.points[mask]

        elif type(geometry) == ogr.Geometry:
            keep_points = clip_funcs.poly_clip(self, geometry)

        return(Cloud(CloudData(keep_points, self.las.header)))

    def filter_z(self,min,max):
        """
        Filters a cloud object based on Z heights
        :param min: Minimum z height in map units
        :param max: Maximum z height in map units
        """

        #Filter condition
        condition = (self.las.points[:,2] > min) &  (self.las.points[:,2] < max)
        self.las.points=self.las.points[condition]

        #reform header
        self.las.x = self.las.points[:, 0]
        self.las.y = self.las.points[:, 1]
        self.las.z = self.las.points[:, 2]

        # TODO consider returning a new cloud object
        self.las.header.min = [np.min(self.las.x), np.min(self.las.y), np.min(self.las.z)]
        self.las.header.max = [np.max(self.las.x), np.max(self.las.y), np.max(self.las.z)]
        self.las.header.count = np.alen(self.las.points)
