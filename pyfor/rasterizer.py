# Functions for rasterizing
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.interpolate import griddata
from scipy.signal import medfilt
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from pyfor import gisexport
from pyfor import filter
from rasterio.transform import from_origin

class Grid:
    """The Grid object is a representation of a point cloud that has been sorted into X and Y dimensional bins. It is \
    not quite a raster yet. A raster has only one value per cell, whereas the Grid object merely sorts all points \
    into their respective cells."""
    # TODO Decide between self.cloud or self.las
    # TODO bw4sz, cell size units?
    def __init__(self, cloud, cell_size):
        """
        Sorts the point cloud into a gridded form such that every point in the las file is assigned a cell coordinate \
        with a resolution equal to cell_size

        :param cloud: The "parent" cloud object.
        :param cell_size: The size of the cell for sorting in the units of the input cloud object.
        :return: Returns a dataframe with sorted x and y with associated bins in a new columns
        """
        self.cloud = cloud
        self.las = self.cloud.las
        self.cell_size = cell_size

        min_x, max_x = self.las.min[0], self.las.max[0]
        min_y, max_y = self.las.min[1], self.las.max[1]

        self.m = int(np.floor((max_y - min_y) / cell_size))
        self.n = int(np.floor((max_x - min_x) / cell_size))

        # Create bins
        bins_x = np.searchsorted(np.linspace(min_x, max_x, self.n), self.las.points["x"])
        bins_y = np.searchsorted(np.linspace(min_y, max_y, self.m), self.las.points["y"])

        self.data = self.las.points
        self.data["bins_x"] = bins_x
        self.data["bins_y"] = bins_y

        self.cells = self.data.groupby(['bins_x', 'bins_y'])

    def raster(self, func, dim):
        """
        Generates an m x n matrix with values as calculated for each cell in func. This is a raw array without \
        missing cells interpolated. See self.interpolate for interpolation methods.

        :param func: A function string, i.e. "max" or a function itself, i.e. np.max. This function must be able to \
        take a 1D array of the given dimension as an input and produce a single value as an output. This single value \
        will become the value of each cell in the array.
        :param dim: The dimension to calculate on as a string, see the column names of self.data for a full list of \
        options
        :return: A 2D numpy array where the value of each cell is the result of the passed function.
        """

        array = self.cells.agg({dim: func}).reset_index().pivot('bins_y', 'bins_x', dim)
        array = np.asarray(array)
        return Raster(array, self)

    def boolean_summary(self, func, dim):
        # TODO Might not be worth its own function...
        """
        Calculates a column in self.data that is a boolean of whether or not that point is the point that corresponds \
        to the function passed. For example, this can be used to create a boolean mask of points that are the minimum \
        z point in their respective cell.

        :param func: The function to calculate on each group.
        :param dim: The dimension of the point cloud as a string (x, y or z)
        """

        mask = self.data.groupby(['bins_x', 'bins_y'])[dim].transform(func) == self.data[dim]
        return mask

    @property
    def empty_cells(self):
        """
        Retrieves the cells with no returns in self.data

        return: An N x 2 numpy array where each row cooresponds to the [y x] coordinate of the empty cell.
        """
        array = self.raster("count", "z").array
        emptys = np.argwhere(np.isnan(array))

        return emptys

    def interpolate(self, func, dim, interp_method="nearest"):
        """
        Interpolates missing cells in the grid. This function uses scipy.griddata as a backend. Please see \
        documentation for that function for more details.

        :param func: The function (or function string) to calculate an array on the gridded data.
        :param dim: The dimension (i.e. column name of self.cells) to cast func onto.
        :param interp_method: The interpolation method call for scipy.griddata, one of any: "nearest", "cubic", \
        "linear"

        :return: An interpolated array.
        """
        # Get points and values that we already have
        cell_values = self.cells[dim].agg(func).reset_index()

        points = cell_values[['bins_x', 'bins_y']].values
        values = cell_values[dim].values

        # https://stackoverflow.com/questions/12864445/numpy-meshgrid-points
        X, Y = np.mgrid[1:self.n+1, 1:self.m+1]

        interp_grid = griddata(points, values, (X, Y), method=interp_method).T

        return Raster(interp_grid, self)

    def metrics(self, func_dict):
        """
        Calculates summary statistics for each grid cell in the Grid.

        :param func_dict: A dictionary containing keys corresponding to the columns of self.data and values that \
        correspond to the functions to be  called on those columns.
        :return: A pandas dataframe with the aggregated metrics.
        """

        return self.cells.agg(func_dict)

    def plot(self, func, cmap="viridis", dim="z", return_plot=False):
        """
        Plots a 2 dimensional canopy height model using the maximum z value in each cell. This is intended for visual \
        checking and not for analysis purposes. See the rasterizer.Grid class for analysis.

        :param func: The function to aggregate the points in the cell.
        :param cmap: A matplotlib color map string.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object.
        """
        # Summarize (i.e. aggregate) on the max z value and reshape the dataframe into a 2d matrix
        plot_mat = self.cells.agg({dim: func}).reset_index().pivot('bins_y', 'bins_x', dim)

        if return_plot == True:
            return(Raster(plot_mat, self).plot(cmap = cmap, return_plot = True))

        Raster(plot_mat, self).plot(return_plot=False)

    def plot3d(self):
        """
        Not yet implemented.
        """
        pass

    def ground_filter(self, num_windows, dh_max, dh_0, interp_method = "nearest"):
        """
        Wrapper call for filter.zhang with convenient defaults.

        Returns a Raster object corresponding to the filtered ground DEM of this particular grid.
        :param type:
        :return:
        """
        # TODO Add functionality for classifying points as ground
        # Get the interpolated DEM array.
        dem_array = filter.zhang(self.interpolate("min", "z").array, num_windows,
                                 dh_max, dh_0, self.cell_size, self, interp_method = interp_method)
        dem = Raster(dem_array, self)

        return dem

    def normalize(self, num_windows, dh_max, dh_0, interp_method="nearest"):
        """
        Returns a new, normalized Grid object.
        :return:
        """

        if self.cloud.normalized == True:
            print("It appears this has already been normalized once. Proceeding with normalization but expect \
            strange results.")

        # Retrieve the DEM
        dem = self.ground_filter(num_windows, dh_max, dh_0, interp_method)

        # Organize the array into a dataframe and merge
        df = pd.DataFrame(dem.array).stack().rename_axis(['bins_y', 'bins_x']).reset_index(name='val')
        #df = pd.merge(self.data, df)

        df = self.data.reset_index().merge(df, how = "left").set_index('index')
        df['z'] = df['z'] - df['val']

        # Initialize new grid object
        ground_grid = Grid(self.cloud, self.cell_size)
        ground_grid.data = df
        ground_grid.cells = ground_grid.data.groupby(['bins_x', 'bins_y'])

        return ground_grid

class Raster:
    def __init__(self, array, grid):
        self.array = array
        self.grid = grid
        self.cell_size = self.grid.cell_size

    @property
    def _affine(self):
        """Constructs the affine transformation, used for plotting and exporting polygons and rasters."""
        affine = from_origin(self.grid.las.min[0], self.grid.las.max[1], self.grid.cell_size, self.grid.cell_size)
        return affine

    def plot(self, cmap = "viridis", return_plot = False):
        """
        Default plotting method for the Raster object.

        :return:
        """
        #TODO implement cmap
        fig = plt.figure()
        ax = fig.add_subplot(111)
        caz = ax.matshow(self.array)
        fig.colorbar(caz)
        fig.gca().invert_yaxis()
        ax.xaxis.tick_bottom()
        ax.set_xticks(np.linspace(0, self.grid.n, 3))
        ax.set_yticks(np.linspace(0, self.grid.m, 3))

        x_ticks, y_ticks = np.rint(np.linspace(self.grid.las.header.min[0], self.grid.las.header.max[0], 3)), \
                           np.rint(np.linspace(self.grid.las.header.min[1], self.grid.las.header.max[1], 3))

        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        if return_plot == True:
            return(fig)

        else:
            plt.show()


    def watershed_seg(self, min_distance = 2, threshold_abs = 2, classify = False):
        """
        Returns the watershed segmentation of the Raster as a geopandas dataframe.

        :param classify: If true, sets the user data of the original point cloud data to the segment ID. The \
        segment ID is an arbitrary identification number generated by the labels function. This can be useful for \
        plotting point clouds where each segment color is unique.
        :return: A geopandas data frame, each record is a crown segment.
        """

        # TODO Not sure if this should be generalized to another class (like grid)
        #if self.grid.cloud.crs == None:
        #    print("Watershed segmentation requires coordinate reference. If your point cloud is referenced in UTM \
        #          lookinto the gisexport.utm_lookup function")
        #    return False

        tops = peak_local_max(self.array, indices = False, min_distance= min_distance, threshold_abs= threshold_abs)
        tops = label(tops)[0]
        labels = watershed(-self.array, tops, mask = self.array)

        if classify == True:
            xy = self.grid.data[["bins_x", "bins_y"]].values
            tree_id = labels[xy[:,1], xy[:,0]]

            # Update the CloudData and Grid objects
            self.grid.las.points["user_data"] = tree_id
            self.grid.data = self.grid.las.points
            self.grid.cells = self.grid.data.groupby(['bins_x', 'bins_y'])


        tops = gisexport.array_to_polygons(labels, self._affine)

        return tops

    def pit_filter(self, kernel_size):
        self.array = medfilt(self.array, kernel_size = kernel_size)

    def write(self, path):
        if self.grid.cloud.crs == None:
            # This should only be the case for older .las files without CRS information
            print("There is no wkt string set for this Grid object, you must manually pass one to the \
            write_raster function. This likely means you are using an older las specification.")
        else:
            print("Raster file written to {}".format(path))
            gisexport.array_to_raster(self.array, self.cell_size, self.grid.las.min[0], self.grid.las.max[1],
                                      self.grid.cloud.crs, path)
