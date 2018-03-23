# An update of the cloudinfo class

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import lasplot


class CloudInfo:
    def __init__(self, las_path):
        self.las = laspy.file.File(las_path)

    def grid(self, cell_size):
        """Sorts the point cloud into a gridded form.

        :param cell_size: The size of the cell for sorting
        :return: Returns a dataframe with sorted x and y with associated bins in a new columns
        """

        min_x, max_x = min(self.las.x), max(self.las.x)
        min_y, max_y = min(self.las.y), max(self.las.y)

        m = int(np.floor((max_y - min_y) / cell_size) + 1)
        n = int(np.floor((max_x - min_x) / cell_size) + 1)

        # Create bins
        bins_x = np.digitize(self.las.x, np.linspace(min(self.las.x), max(self.las.x), n + 1))
        bins_y = np.digitize(self.las.y, np.linspace(min(self.las.y), max(self.las.y), m + 1))

        # Add bins and las data to a new dataframe
        df = pd.DataFrame({'x': self.las.x, 'y': self.las.y, 'z': self.las.z, 'bins_x': bins_x, 'bins_y': bins_y})
        return(df)

    def plot(self, cell_size = 1):
        # Group by the x and y grid cells
        gridded_df = self.grid(cell_size)
        group_df = gridded_df[['bins_x', 'bins_y', 'z']].groupby(['bins_x', 'bins_y'])

        # Summarize (i.e. aggregate) on the max z value and reshape the dataframe into a 2d matrix
        plot_mat = group_df.agg({'z': 'max'}).reset_index().pivot('bins_y', 'bins_x', 'z')

        # Plot the matrix, and invert the y axis to orient the 'image' appropriately
        plt.matshow(plot_mat)
        plt.gca().invert_yaxis()

        # Show the matrix image
        plt.show()

    def plot3d(self, point_size = 1, cmap = 'Spectral_r'):
        # Generate the coordinates
        coordinates = np.stack([self.las.x, self.las.y, self.las.z], axis = 1)

        # Start Qt app and widget
        pg.mkQApp()
        view = gl.GLViewWidget()

        # Normalize Z to 0-1 space
        z = np.copy(coordinates[:,2])
        z = (z - min(z)) / (max(z) - min(z))

        # Get matplotlib color maps
        cmap = cm.get_cmap('Spectral_r')
        colors = cmap(z)

        # Create the points, change to opaque, set size to 1
        points = gl.GLScatterPlotItem(pos = coordinates, color = colors)
        points.setGLOptions('opaque')
        points.setData(size = np.repeat(point_size, len(coordinates)))

        # Add points to the viewer
        view.addItem(points)

        # Center on the aritgmetic mean of the point cloud and display
        center = np.mean(coordinates, axis = 0)
        view.opts['center'] = pg.Vector(center[0], center[1], center[2])
        view.show()

