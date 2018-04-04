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

        # Assign useful shortcuts
        # TODO expand
        self.points = points
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = self.points[:, 2]

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
            points = np.stack((las.x, las.y, las.z, las.intensity, las.flag_byte,
                                    las.classification, las.scan_angle_rank, las.user_data,
                                    las.pt_src_id), axis=1)
            header = las.header

            # Toss out the laspy object in favor of CloudData
            self.las = CloudData(points, header)
        elif type(las) == CloudData:
            self.las = las
        else:
            print("Object type not supported, please input either a las file path or a CloudData object.")

    def grid(self, cell_size):
        return(rasterizer.Grid(self, cell_size))

    def plot(self, cell_size = 1, return_plot = False):
        """
        Plots a 2 dimensional canopy height model using the maximum z value in each cell. This is intended for visual
        checking and not for analysis purposes. See the rasterizer.Grid class for analysis.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object.
        """
        # Group by the x and y grid cells
        gridded_df = self.grid(cell_size).data
        group_df = gridded_df[['bins_x', 'bins_y', 'z']].groupby(['bins_x', 'bins_y'])

        # Summarize (i.e. aggregate) on the max z value and reshape the dataframe into a 2d matrix
        plot_mat = group_df.agg({'z': 'max'}).reset_index().pivot('bins_y', 'bins_x', 'z')

        # Plot the matrix, and invert the y axis to orient the 'image' appropriately
        plt.matshow(plot_mat)
        plt.gca().invert_yaxis()

        #TODO Fix plot axes
        if return_plot:
            return(plt)
        else:
            # Show the matrix image
            plt.show()

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
