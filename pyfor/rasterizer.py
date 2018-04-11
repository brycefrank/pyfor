# Functions for rasterizing
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pyfor import gisexport
from pyfor import metrics2
from pyfor import filter

# TODO: refactor any grouped dataframe to "cells"

class Grid:
    """The grid object constructs a grid from a given Cloud object and cell_size and contains functions useful
    for manipulating rasterized data."""
    # TODO Decide between self.cloud or self.las
    # TODO bw4sz, cell size units?
    def __init__(self, cloud, cell_size):
        """
        Sorts the point cloud into a gridded form such that every point in the las file is assigned a cell coordinate
        with a resolution equal to cell_size
        :param cell_size: The size of the cell for sorting in the units of the input cloud object
        :param indices: The indices of self.points to plot
        :return: Returns a dataframe with sorted x and y with associated bins in a new columns
        """
        self.las = cloud.las
        self.cell_size = cell_size

        # TODO Need to update headers when new cloud is constructed
        min_x, max_x = self.las.header.min[0], self.las.header.max[0]
        min_y, max_y = self.las.header.min[1], self.las.header.max[1]

        self.m = int(np.floor((max_y - min_y) / cell_size))
        self.n = int(np.floor((max_x - min_x) / cell_size))

        # Create bins
        bins_x = np.searchsorted(np.linspace(min_x, max_x, self.n), self.las.x)
        bins_y = np.searchsorted(np.linspace(min_y, max_y, self.m), self.las.y)

        # Add bins and las data to a new dataframe
        self.data = pd.DataFrame({'x': self.las.x, 'y': self.las.y, 'z': self.las.z,
                           'bins_x': bins_x, 'bins_y': bins_y})

        self.cells = self.data.groupby(['bins_x', 'bins_y'])

    def array(self, func, dim):
        """
        Generates an m x n matrix with values as calculated for each cell in func. This is a raw
        array without missing cells interpolated. See self.interpolate for interpolation methods.

        :param func: A function string, i.e. "max", a function itself, i.e. max, or a Metrics object. This function
        must be able to take an array as an input and produce a single value as an output. This single value will
        become the value of each cell in the array.
        :param dim: The dimension to calculate on as a string, see the column names of self.data for a full list of
        options
        :return: A 2D numpy array where the value of each cell is the result of the passed function.
        """
        array = self.cells.agg({dim: func}).reset_index().pivot('bins_x', 'bins_y', dim)
        array = np.array(array)
        return(array)

    def boolean_summary(self, func, dim):
        # TODO Might not be worth its own function...
        """
        Calculates a column in self.data that is a boolean of whether
        or not that point is the point that corresponds to the function passed.

        For example, this can be used to create a boolean mask of points that
        are the minimum z point in their respective cell.

        :param func: The function to calculate on each group.
        :param dim: The dimension of the point cloud as a string (x, y or z)
        """

        mask = self.data.groupby(['bins_x', 'bins_y'])[dim].transform(func) == self.data[dim]
        return(mask)

    @property
    def empty_cells(self):
        # TODO Very slow.
        """
        Retrieves the cells with no returns in self.data
        """
        array = self.array("count", "z")
        emptys = np.argwhere(np.isnan((array)))

        return(emptys)

    def _interpolate(self, func, dim, interp_method="nearest"):
        """
        # TODO Decide on return type, matrix or append to self.data? This decision can be made
        after more IO stuff is written. It should probably return a saveable / plottable
        raster object of some sort. Should I make a raster class, or just flesh out grid?

        Interpolates missing cells in the grid.
        """
        # Get points and values that we already have
        cell_values = self.cells[dim].agg(func).reset_index()

        points = cell_values[['bins_x', 'bins_y']].values
        values = cell_values[dim].values

        # https://stackoverflow.com/questions/12864445/numpy-meshgrid-points
        X, Y = np.mgrid[1:self.n+1, 1:self.m+1]

        interp_grid = griddata(points, values, (X, Y), method = interp_method).T

        return(interp_grid)

    def metrics(self, func_string, dim):
        """
        Calculates summary statistics for each grid cell in the Grid.

        :return:
        """

        # We have a grouped dataframe (we will group all of the data for now:
        cells = self.data.groupby(['bins_x', 'bins_y'])[dim]

        return(cells.agg(func_string))

    def plot(self, func, cmap ="viridis", dim = "z", return_plot = False):
        """
        Plots a 2 dimensional canopy height model using the maximum z value in each cell. This is intended for visual
        checking and not for analysis purposes. See the rasterizer.Grid class for analysis.

        :param func: The function to aggregate the points in the cell.
        :param cmap: A matplotlib color map string.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object.
        """
        # Summarize (i.e. aggregate) on the max z value and reshape the dataframe into a 2d matrix
        plot_mat = self.cells.agg({dim: func}).reset_index().pivot('bins_y', 'bins_x', 'z')

        # Plot the matrix, and invert the y axis to orient the 'image' appropriately
        plt.matshow(plot_mat, cmap)
        plt.gca().invert_yaxis()

        # TODO Fix plot axes
        if return_plot:
            return(plt)
        else:
            # Show the matrix image
            plt.show()

    def ground_filter(self):
        """
        Wrapper call for filter.zhang with convenient defaults.
        :param type:
        :return:
        """
        # Get the interpolated DEM array.
        dem_array = filter.zhang(self._interpolate("min", "z"), 3, 1.5, 0.5, self.cell_size, self)
        dem_array = Raster(dem_array)

        return(dem_array)

    def write_raster(self, path, func, dim, wkt = None):
        if self.las.wkt == None:
            # This should only be the case for older .las files without CRS information
            print("There is no wkt string set for this Grid object, you must manually pass one to the \
            write_raster function. This likely means you are using an older las specification.")
        else:
            write_array = self.array(func, dim)
            print("Raster file written to {}".format(path))
            gisexport.array_to_raster(write_array, self.cell_size, self.las.header.min[0], self.las.header.max[1], path)


class Raster:
    def __init__(self, array, crs = None, cell_size = 1):
        self.array = array
        pass

    def plot(self):
        plt.matshow(self.array)
        plt.show()

    def write_raster(self):
        if self.crs == None:
            # This should only be the case for older .las files without CRS information
            print("There is no wkt string set for this Grid object, you must manually pass one to the \
            write_raster function. This likely means you are using an older las specification.")
        else:
            print("Raster file written to {}".format(path))
            gisexport.array_to_raster(write_array, self.cell_size, self.las.header.min[0], self.las.header.max[1], path)
