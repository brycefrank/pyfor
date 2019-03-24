import os
import laspy
import numpy as np
import pyfor
import geopandas as gpd
import pandas as pd
import laxpy
import pyproj
from shapely.geometry import Polygon

class CloudDataFrame(gpd.GeoDataFrame):
    """
    Implements a data frame structure for processing and managing multiple :class:`.Cloud` objects. It is recommended \
    to initialize using the :func:`.from_dir` function.
    """
    def __init__(self, *args, **kwargs):
        super(CloudDataFrame, self).__init__(*args, **kwargs)
        self.n_threads = 1
        self.tiles = None
        self.crs = None

        if "bounding_box" in self.columns.values:
            self.set_geometry("bounding_box", inplace=True)
            self.tiles = self['bounding_box'].values

    @classmethod
    def _from_dir(cls, las_dir, n_jobs=1, get_bounding_boxes=True):
        """
        Wrapped function for producing a CloudDataFrame from a directory of las files.
        :param las_dir: A directory of .las or .laz files.
        :param n_jobs: The number of threads used to construct information about the CloudDataFrame.
        :param get_bounding_boxes: If True, builds the bounding boxes for each las tile by manually reading in
        the file and computing the bounding box. For very large collections this may be computationally costly, and
        can be set to False.
        :return:
        """
        las_path_init = [[os.path.join(root, file) for file in files] for root, dirs, files in os.walk(las_dir)][0]
        las_path_init = [las_path for las_path in las_path_init if las_path.endswith('.las')]
        cdf = CloudDataFrame({'las_path': las_path_init})
        cdf.n_threads = n_jobs

        if get_bounding_boxes == True:
            cdf._build_polygons()

        return(cdf)

    def set_index(self, *args):
        return CloudDataFrame(super(CloudDataFrame, self).set_index(*args))

    @property
    def bounding_box(self):
        """Retrieves the bounding box for the entire collection. As a tuple (minx, muny, maxx, maxy)"""
        minx, miny, maxx, maxy = [i.bounds[0] for i in self['bounding_box']], [i.bounds[1] for i in self['bounding_box']], \
                                 [i.bounds[2] for i in self['bounding_box']], [i.bounds[3] for i in self['bounding_box']]
        col_bbox = np.min(minx), np.min(miny), np.max(maxx), np.max(maxy)
        return col_bbox

    def map_poly(self, las_path, polygon):
        las = laxpy.IndexedLAS(las_path)
        las.map_polygon(polygon)
        print(len(las.points))
        return las.points

    def construct_tile(self, func, tile, indexed):
        """
        For a given tile, clips points from intersecting las files and loads as Cloud object.
        """
        for i, las_file in enumerate(self._get_parents(tile)['las_path']):
            if indexed == True:
                try:
                    las = laxpy.IndexedLAS(las_file)
                    las.map_polygon(tile)
                except ValueError as e:
                    print(e)
                    return 0
            else:
                las = las_file

            if i == 0:
                out_pc = pyfor.cloud.Cloud(las)
            else:
                out_pc.data._append(pyfor.cloud.Cloud(las).data)

        try:
            out_pc = out_pc.clip(tile)
            out_pc.crs = self.crs
            func(out_pc, tile)
        except KeyError:
            pass

    def par_apply(self, func, indexed=True, by_file=False, *args):
        """
        Apply a function to the point cloud described by each tile in `self.tiles` such that the first argument of the
        function is a :class:`pyfor.cloud.Cloud` object. This is achieved via :class:`joblib.Parallel` and :func:`joblib.delayed`.

        Interested in applying a function to buffered point clouds? That should be implemented with a retiling operation first
        and then brought here.

        :param func: The user defined function, must accept as its first argument a :class`pyfor.cloud.Cloud` object.
        :param *args: Further arguments to `func`
        """

        from joblib import Parallel, delayed

        if by_file:
            return Parallel(n_jobs=self.n_threads)(delayed(func)(las_path) for las_path in self['las_path'])
        else:
            Parallel(n_jobs=self.n_threads)(delayed(self.construct_tile)(func, tile, indexed) for tile in self.tiles)


    def retile_buffer(self):
        """
        A basic retiling operation that buffers the current `self.tiles` using a square buffer.
        :return:
        """

        pass

    def retile_raster(self, cell_size, original_tile_size, buffer=0):
        """
        A retiling operation that creates raster-compatible sized tiles. Important for creating project-level rasters.
        Changes `self.tiles` **in place**

        :param buffer: Buffer the raster-compatible tiles.
        """

        retiler = Retiler(self)
        self.tiles = retiler.retile_raster(cell_size, original_tile_size, buffer)

    def _index_las(self, las_path):
        """
        Checks if an equivalent `.lax` file exists. If so, creates a laxpy.IndexedLAS object, otherwise an error is thrown.
        """
        lax_path = las_path[:-1] + 'x'

        if os.path.isfile(lax_path):
            return laxpy.IndexedLAS(las_path)
        else:
            raise FileNotFoundError('There is no equivalent .lax file for this .las file.')

    def _get_bounding_box(self, las_path):
        """
        Vectorized function to get a bounding box from an individual las path.
        :param las_path:
        :return:
        """
        # segmentation of point clouds
        pc = laspy.file.File(las_path)
        min_x, max_x = pc.header.min[0], pc.header.max[0]
        min_y, max_y = pc.header.min[1], pc.header.max[1]
        return((min_x, max_x, min_y, max_y))

    def _build_polygons(self):
        """Builds the shapely polygons of the bounding boxes and adds them to self.data"""
        from shapely.geometry import Polygon

        bboxes = [self._get_bounding_box(las_path) for las_path in self['las_path']]
        print(bboxes)
        self["bounding_box"] = [Polygon(((bbox[0], bbox[2]), (bbox[1], bbox[2]),
                                           (bbox[1], bbox[3]), (bbox[0], bbox[3]))) for bbox in bboxes]
        self.set_geometry("bounding_box", inplace = True)
        self.tiles = self['bounding_box'].values

    def _get_parents(self, polygon):
        """
        For a given input polygon, finds the files whose bounding boxes intersect with that polygon.

        :param polygon:
        :return: A GeoDataFrame of intersecting file bounding boxes.
        """
        return self[self['bounding_box'].intersects(polygon)]

    def plot(self, **kwargs):
        """
        Plots the bounding boxes of the Cloud objects.

        :param **kwargs: Keyword arguments to :meth:`geopandas.GeoDataFrame.plot`.
        """
        plot = super(CloudDataFrame, self).plot(**kwargs)
        plot.figure.show()

    def create_index(self):
        """
        For each file in the collection, creates `.lax` files for spatial indexing using the default values.
        """
        for las_path in self['las_path']:
            laxpy.file.init_lax(las_path)

    def clip(self):
        # TODO 0.3.3 clip was a mess, should be easy to leverage existing tiling functions for this.
        pass


    def standard_metrics(self, heightbreak, index=None):
        """
        Retrieves a set of 29 standard metrics, including height percentiles and other summaries.
        :param index: An iterable of indices to set as the output dataframe index.
        :return: A pandas dataframe of standard metrics.
        """
        from pyfor.metrics import standard_metrics
        get_metrics = lambda las_path: standard_metrics(pyfor.cloud.Cloud(las_path).data.points,
                                                        heightbreak=heightbreak)
        metrics = pd.concat(self.par_apply(get_metrics, by_file=True), sort=False)

        if index:
            metrics.index = index

        return metrics

class Retiler:
    def __init__(self, cdf):
        """
        Retiles a CloudDataFrame. Generally used to create tiles such that rasters generated from tiles are properly aligned.
        """
        self.cdf = cdf


    def _square_buffer(self, polygon, buffer):
        """
        A simple square buffer that expands a square by a given buffer distance.

        :param polygon: A shapely polygon.
        :param buffer: The buffer distance.
        :return: A buffered shapely polgon
        """

        minx, miny, maxx, maxy = polygon.bounds
        n_minx, n_miny, n_maxx, n_maxy = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer

        buffered_poly = Polygon([[n_minx, n_miny],
                                 [n_minx, n_maxy],
                                 [n_maxx, n_maxy],
                                 [n_maxx, n_miny]])
        return buffered_poly

    def retile_raster(self, target_cell_size, original_tile_size, buffer = 0):
        """
        Creates a retiling grid for a specified target cell size. This creates a list of polygons such that if a raster
        is constructed from a polygon it will exactly fit inside given the specified target cell size. Useful for creating
        project level rasters.

        :param target_cell_size: The desired output cell size
        :param original_tile_size: The original tile size of the project
        :param buffer: The distance to buffer each new tile to prevent edge effects.
        :return: A list of shapely polygons that correspond to the new grid.
        """
        from shapely.geometry import Polygon


        bottom, left = self.cdf.bounding_box[1], self.cdf.bounding_box[0]
        top, right = self.cdf.bounding_box[3], self.cdf.bounding_box[2]

        new_tile_size = np.ceil(original_tile_size / target_cell_size) * target_cell_size

        project_width = right - left
        project_height = top - bottom

        num_x = int(np.ceil(project_width / new_tile_size))
        num_y = int(np.ceil(project_height / new_tile_size))

        new_tiles = []
        for i in range(num_x):
            for j in range(num_y):
                # Create geometry
                tile_left, tile_bottom = left + i * new_tile_size, bottom + j * new_tile_size

                new_tile = Polygon([
                    [tile_left, tile_bottom], #bl
                    [tile_left, tile_bottom + new_tile_size], #tl
                    [tile_left + new_tile_size, tile_bottom + new_tile_size], #tr
                    [tile_left + new_tile_size, tile_bottom]]) #br

                if buffer > 0:
                    new_tile = self._square_buffer(new_tile, buffer)

                # Only append if there are any original tiles touching
                if len(self.cdf._get_parents(new_tile)) > 0:
                    new_tiles.append(new_tile)

        return new_tiles

    def retile_buffer(self, tiles, buffer):
        """
        A simple buffering operation.

        :return: A list of buffered shapely polygons.
        """

        return [self._square_buffer(tile, buffer) for tile in tiles]


    def retile_quadrant(self, tiles):
        """
        Splits input tiles into quadrants. An efficient retiling method when particular retiling geometries are not
        necessary.
        """

        new_tiles = []
        for tile in tiles:
            # Build the quadrant geometries, this is defined by the following six values
            x0, y0 = tile[0], tile[1]
            x2, y2 = tile[2], tile[3]
            x1 = ((tile[2] - tile[0]) / 2) + x0
            y1 = ((tile[3] - tile[1]) / 2) + y0

            # Create the geometries
            bottom_left = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
            bottom_right = Polygon([(x1, y0), (x1, y1), (x2, y1), (x2, y0)])
            top_left = Polygon([(x0, y1), (x0, y2), (x1, y2), (x1, y1)])
            top_right = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

            quadrants = [bottom_left, bottom_right, top_left, top_right]
            new_tiles.append(quadrants)

        return [quadrant for sublist in new_tiles for quadrant in sublist]

def from_dir(las_dir, **kwargs):
    """
    Constructs a CloudDataFrame from a directory of las files.

    :param las_dir: The directory of las files.
    :return: A CloudDataFrame constructed from the directory of las files.
    """

    return CloudDataFrame._from_dir(las_dir, **kwargs)

