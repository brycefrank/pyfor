# Functions for rasterizing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyfor import gisexport

class Grid:
    """The Grid object is a representation of a point cloud that has been sorted into X and Y dimensional bins. From \
    the Grid object we can derive other useful products, most importantly, :class:`.Raster` objects.
    """

    def __init__(self, cloud, cell_size):
        """
        Upon initialization, the parent cloud object's :attr:`data.points` attribute is sorted into bins in place. \ The
        columns 'bins_x' and 'bins_y' are appended. Other useful information, such as the resolution, number of rows \
        and columns are also stored.

        :param cloud: The "parent" cloud object.
        :param cell_size: The size of the cell for sorting in the units of the input cloud object.
        """

        self.cloud = cloud
        self.cell_size = cell_size

        min_x, max_x = self.cloud.data.min[0], self.cloud.data.max[0]
        min_y, max_y = self.cloud.data.min[1], self.cloud.data.max[1]

        self.m = int(np.ceil((max_y - min_y) / cell_size))
        self.n = int(np.ceil((max_x - min_x) / cell_size))

        x_edges = np.arange(min_x, max_x, self.cell_size)
        y_edges = np.arange(min_y, max_y, self.cell_size)

        bins_x = np.searchsorted(x_edges,   self.cloud.data.points['x'], side='right') - 1
        bins_y = np.searchsorted(-y_edges, -self.cloud.data.points['y'], side='left', sorter=(-y_edges).argsort())

        self.cloud.data.points["bins_x"] = bins_x
        self.cloud.data.points["bins_y"] = bins_y

        self.cells = self.cloud.data.points.groupby(['bins_x', 'bins_y'])

    def _update(self):
        self.cloud.data._update()
        self.__init__(self.cloud, self.cell_size)

    def raster(self, func, dim, **kwargs):
        """
        Generates an m x n matrix with values as calculated for each cell in func. This is a raw array without \
        missing cells interpolated. See self.interpolate for interpolation methods.

        :param func: A function string, i.e. "max" or a function itself, i.e. :func:`np.max`. This function must be \
        able to take a 1D array of the given dimension as an input and produce a single value as an output. This \
        single value will become the value of each cell in the array.
        :param dim: A dimension to calculate on.
        :return: A 2D numpy array where the value of each cell is the result of the passed function.
        """
        bin_summary = self.cells.agg({dim: func}, **kwargs).reset_index()
        array = np.full((self.m, self.n), np.nan)
        array[bin_summary["bins_y"], bin_summary["bins_x"]] = bin_summary[dim]
        return Raster(array, self)

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
        from scipy.interpolate import griddata
        # Get points and values that we already have
        cell_values = self.cells[dim].agg(func).reset_index()

        points = cell_values[['bins_x', 'bins_y']].values
        values = cell_values[dim].values

        # https://stackoverflow.com/questions/12864445/numpy-meshgrid-points
        # TODO assumes a raster occupies a square/rectangular space. Is it possible to not assume this and increase performance?
        X, Y = np.mgrid[1:self.n+1, 1:self.m+1]

        # TODO generally a slow approach
        interp_grid = griddata(points, values, (X, Y), method=interp_method).T

        return Raster(interp_grid, self)

    def metrics(self, func_dict, as_raster = False):
        """
        Calculates summary statistics for each grid cell in the Grid.

        :param func_dict: A dictionary containing keys corresponding to the columns of self.data and values that \
        correspond to the functions to be  called on those columns.
        :return: A pandas dataframe with the aggregated metrics.
        """

        # Aggregate on the function
        aggregate = self.cells.agg(func_dict)
        if as_raster == False:
            return aggregate
        else:
            rasters = []
            for column in aggregate:
                array = np.asarray(aggregate[column].reset_index().pivot('bins_y', 'bins_x'))
                raster = Raster(array, self)
                rasters.append(raster)
            # Get list of dimension names
            dims = [tup[0] for tup in list(aggregate)]
            # Get list of metric names
            metrics = [tup[1] for tup in list(aggregate)]
            return pd.DataFrame({'dim': dims, 'metric': metrics, 'raster': rasters}).set_index(['dim', 'metric'])

class ImportedGrid(Grid):
    """
    ImportedGrid is used to normalize a parent cloud object with an arbitrary raster file.
    """

    def __init__(self, path, cloud):
        import rasterio
        self.in_raster = rasterio.open(path)

        # Check cell size
        cell_size_x, cell_size_y = self.in_raster.transform[0], abs(self.in_raster.transform[4])
        if cell_size_x != cell_size_y:
            print('Cell sizes not equal of input raster, not supported.')
            raise ValueError
        else:
            cell_size = cell_size_x

        self.cloud = cloud
        self.cell_size = cell_size

        min_x, max_x = self.in_raster.bounds[0], self.in_raster.bounds[2]
        min_y, max_y = self.in_raster.bounds[1], self.in_raster.bounds[3]

        self.m = self.in_raster.height
        self.n = self.in_raster.width


        #min_x, max_x = self.cloud.data.min[0], self.cloud.data.max[0]
        #min_y, max_y = self.cloud.data.min[1], self.cloud.data.max[1]

        #self.m = int(np.floor((max_y - min_y) / cell_size))
        #self.n = int(np.floor((max_x - min_x) / cell_size))

        # Create bins
        bins_x = np.searchsorted(np.linspace(min_x, max_x, self.n), self.cloud.data.points["x"])
        bins_y = np.searchsorted(np.linspace(min_y, max_y, self.m), self.cloud.data.points["y"])

        self.cloud.data.points["bins_x"] = bins_x
        self.cloud.data.points["bins_y"] = bins_y
        self.cells = self.cloud.data.points.groupby(['bins_x', 'bins_y'])

    def _update(self):
        self.cloud.data._update()


class Raster:
    def __init__(self, array, grid):
        from rasterio.transform import from_origin

        self.grid = grid
        self.cell_size = self.grid.cell_size

        self.array = array
        self._affine = from_origin(self.grid.cloud.data.min[0], self.grid.cloud.data.max[1], self.grid.cell_size, self.grid.cell_size)

    @property
    def _convex_hull_mask(self):
        """
        Calculates an m x n boolean numpy array where the value of each cell represents whether or not that cell lies \ 
        within the convex hull of the "parent" cloud object. This is used when plotting and writing interpolated \
        rasters.
        :return: 
        """
        import fiona
        import rasterio
        from rasterio.mask import mask
        # FIXME broken in 0.3.2, but really needs a rewrite anyway. May be best to remove fiona as a dependency
        # Also fix the test in the testing suite
        self.grid.cloud.convex_hull.to_file("temp.shp")
        with fiona.open("temp.shp", "r") as shapefile:
            features = [feature["geometry"] for feature in shapefile]

        self.write("temp.tif")
        with rasterio.open("temp.tif") as rast:
            out_image = mask(rast, features, nodata = np.nan, crop=True)

        return out_image[0].data


    def force_extent(self, bbox):
        """
        Sets `self._affine` and `self.array` to a forced bounding box. Useful for trimming edges off of rasters when
        processing buffered tiles. This operation is done in place.

        :param bbox: Coordinates of output raster as a tuple (min_x, max_x, min_y, max_y)
        """
        from rasterio.transform import from_origin
        new_left, new_right, new_bot, new_top = bbox

        m, n = self.array.shape[0], self.array.shape[1]

        # Maniupulate the array to fit the new affine transformation
        old_left, old_top = self.grid.cloud.data.min[0], self.grid.cloud.data.max[1]
        old_right, old_bot = old_left + n * self.grid.cell_size, old_top - m * self.grid.cell_size

        left_diff, top_diff, right_diff, bot_diff = old_left - new_left, old_top - new_top, old_right - new_right, old_bot - new_bot
        left_diff, top_diff, right_diff, bot_diff = int(np.rint(left_diff / self.cell_size)), int(np.rint(top_diff / self.cell_size)), \
                                                    int(np.rint(right_diff / self.cell_size)), int(np.rint(bot_diff / self.cell_size))

        if (left_diff > 0 ):
            # bbox left is outside of raster left, we need to add columns of nans
            emptys = np.empty((m, left_diff))
            emptys[:] = np.nan
            self.array = np.insert(self.array, 0, emptys, axis=1)
        elif left_diff !=0:
            # bbox left is inside of raster left, we need to remove left diff columns
            self.array = self.array[:, abs(left_diff):]

        if (top_diff < 0):
            # bbox top is outside of raster top, we need to add rows of nans
            emptys = np.empty((abs(top_diff), self.array.shape[1]))
            emptys[:] = np.nan
            self.array = np.insert(self.array, 0, emptys, axis=0)
        elif top_diff !=0:
            # bbox top is inside of raster top, we need to remove rows of nans
            self.array = self.array[abs(top_diff):, :]

        if (right_diff < 0):
            # bbox right is outside of raster right, we need to add columns of nans
            emptys = np.empty((self.array.shape[0], abs(right_diff)))
            emptys[:] = np.nan
            self.array = np.append(self.array, emptys, axis=1)
        elif right_diff !=0:
            # bbox right is inside raster right, we need to remove columns
            self.array = self.array[:, :-right_diff]

        if (bot_diff > 0):
            # bbox bottom is outside of raster bottom, we need to add columns
            emptys = np.empty((n, abs(bot_diff)))
            emptys[:] = np.nan
            self.array = np.append(self.array, emptys, axis=0)
        elif bot_diff != 0:
            # bbox bottom is inside of raster bottom, we need to remove columns
            self.array = self.array[:bot_diff,:]


        # Handle the affine transformation
        new_affine = from_origin(old_left + ((-left_diff) * self.grid.cell_size), old_top - (top_diff * self.grid.cell_size), self.grid.cell_size, self.grid.cell_size)
        self._affine = new_affine

    def remove_sparse_cells(self, min_points):
        """
        Sets sparse cells, i.e. those that have < `min_points` to nan **in place**. This improves the reliability of edge-cells
        when handling multiple tile rasters.

        :return:
        """

        with np.errstate(invalid='ignore'):
            count_array = self.grid.raster("count", "z").array
            self.array[count_array < min_points] = np.nan

    def plot(self, cmap = "viridis", block = False, return_plot = False):
        """
        Default plotting method for the Raster object.
        """

        #TODO implement cmap
        fig = plt.figure()
        ax = fig.add_subplot(111)
        caz = ax.matshow(self.array)
        fig.colorbar(caz)
        ax.xaxis.tick_bottom()
        ax.set_xticks(np.linspace(0, self.grid.n, 3))
        ax.set_yticks(np.flip(np.linspace(0, self.grid.m, 3)))

        x_ticks, y_ticks = np.rint(np.linspace(self.grid.cloud.data.min[0], self.grid.cloud.data.max[0], 3)), \
                           np.rint(np.linspace(self.grid.cloud.data.min[1], self.grid.cloud.data.max[1], 3))

        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        if return_plot == True:
            return(ax)

        else:
            plt.show(block = block)

    def local_maxima(self, min_distance=2, threshold_abs=2, as_coordinates=False):
        """
        Returns a new Raster object with tops detected using a local maxima filtering method. See
        skimage.feature.peak_local_maxima for more information on the filter.

        :param min_distance:
        :param threshold_abs:
        :param multi_top: If multi_top is true, a top can consist of more than one pixel.
        :param as_coordinates: Not yet implemented
        :return:
        """
        from skimage.feature import peak_local_max, corner_peaks
        from scipy.ndimage import label
        tops = peak_local_max(self.array, indices=False, min_distance=min_distance, threshold_abs=threshold_abs)
        tops = label(tops)[0]

        # TODO Had to take out corner filter to remove duplicate tops.
        tops_raster = DetectedTops(tops, self.grid, self)
        return(tops_raster)


    def watershed_seg(self, min_distance=2, threshold_abs=2, classify=False):
        """
        Returns the watershed segmentation of the Raster as a geopandas dataframe.

        :param min_distance: The minimum distance between local height maxima in the same units as the input point \
        cloud.
        :param threshold_abs: The minimum threshold needed to be called a peak in peak_local_max.
        :param classify: If true, sets the `user_data` of the original point cloud data to the segment ID. The \
        segment ID is an arbitrary identification number generated by the labels function. This can be useful for \
        plotting point clouds where each segment color is unique.
        :return: A geopandas data frame, each record is a crown segment.
        """

        return CrownSegments(self.array, self.grid, min_distance=min_distance, threshold_abs=threshold_abs)

        #if classify == True:
        #    xy = self.grid.cloud.data.points[["bins_x", "bins_y"]].values
        #    tree_id = labels[xy[:, 1], xy[:, 0]]

        #    # Update the CloudData and Grid objects
        #    self.grid.cloud.data.points["user_data"] = tree_id
        #   self.grid.cells = self.grid.cloud.data.points.groupby(['bins_x', 'bins_y'])

    def pit_filter(self, kernel_size):
        """
        Filters pits in the raster. Intended for use with canopy height models (i.e. grid(0.5).interpolate("max", "z").
        This function modifies the raster array **in place**.
        
        :param kernel_size: The size of the kernel window to pass over the array. For example 3 -> 3x3 kernel window.
        """
        from scipy.signal import medfilt2d
        self.array = medfilt2d(self.array, kernel_size=kernel_size)

    def write(self, path):
        """
        Writes the raster to a geotiff. Requires the Cloud.crs attribute to be filled by a projection string (ideally \
        wkt or proj4).
        
        :param path: The path to write to.
        """

        if not self.grid.cloud.crs:
            from warnings import warn
            warn('No coordinate reference system defined. Please set the .crs attribute of the Cloud object.', UserWarning)

        gisexport.array_to_raster(self.array, self._affine, self.grid.cloud.crs, path)


class DetectedTops(Raster):
    """
    This class is for visualization of detected tops with a raster object. Generally created internally via
    Raster.local_maxima
    """

    def __init__(self, array, grid, chm):
        super().__init__(array, grid)
        self.chm = chm

    @property
    def points(self):
        """
        Returns an Nx2 numpy array of detected top locations projected to the input Cloud space.
        """
        from pyfor.gisexport import project_indices
        return project_indices(np.stack(np.where(self.array > 0), axis=1), self)

    @property
    def tops_binary(self):
        """
        :return: A binary array, where 1 indicates a detected top, 0 otherwise
        """
        return (self.array > 0).astype(np.int)

    def plot(self):
        """
        Plots the detected tops against the original input raster.
        # https://matplotlib.org/gallery/images_contours_and_fields/image_transparency_blend.html
        """

        fig, ax = plt.subplots()
        caz = ax.matshow(self.chm.array)
        fig.colorbar(caz)


        # TODO I might not need to repeat this code from raster
        ax.xaxis.tick_bottom()
        ax.set_xticks(np.linspace(0, self.grid.n, 3))
        ax.set_yticks(np.linspace(0, self.grid.m, 3))

        x_ticks, y_ticks = np.rint(np.linspace(self.grid.cloud.data.min[0], self.grid.cloud.data.max[0], 3)), \
                           np.rint(np.linspace(self.grid.cloud.data.min[1], self.grid.cloud.data.max[1], 3))

        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(reversed(y_ticks))

        container = np.zeros((self.grid.m, self.grid.n, 4))
        container[:, :, 0][self.tops_binary >0] = 1
        container[:, :, 3][self.tops_binary >0] = 1
        ax.imshow(container)


class CrownSegments(Raster):
    """
    This class is for visualization of detected crown segments with a raster object.
    """

    def __init__(self, array, grid, min_distance, threshold_abs):
        from skimage.morphology import watershed
        super().__init__(array, grid)
        watershed_array = self.array
        tops = self.local_maxima(min_distance=min_distance, threshold_abs=threshold_abs).array
        self.labels = watershed(-watershed_array, tops, mask=watershed_array)

    @property
    def segments(self):
        return gisexport.array_to_polygons(self.labels, affine=self._affine)

    def plot(self):
        from matplotlib.collections import PatchCollection
        from descartes.patch import PolygonPatch

        geoms = self.segments['geometry'].translate(xoff=-0.5, yoff=-0.5).values

        fig, ax = plt.subplots()
        ax.imshow(self.array)
        ax.add_collection(PatchCollection([PolygonPatch(poly) for poly in geoms], facecolors=(1,0,0,0), edgecolors='#e8e8e8'))
        plt.xlim((0, self.array.shape[1]))
        plt.ylim((0, self.array.shape[0]))
        ax.invert_yaxis()
        plt.show()
