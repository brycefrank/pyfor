import os
import laspy
import numpy as np
import pyfor
import geopandas as gpd
import pandas as pd
import laxpy
import pyproj


class CloudDataFrame(gpd.GeoDataFrame):
    """
    Implements a data frame structure for processing and managing multiple :class:`.Cloud` objects. It is recommended \
    to initialize using the :func:`.from_dir` function.
    """
    def __init__(self, *args, **kwargs):
        super(CloudDataFrame, self).__init__(*args, **kwargs)
        self.n_threads = 1

        if "bounding_box" in self.columns.values:
            self.set_geometry("bounding_box", inplace=True)

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
        las_path_init = [las_path for las_path in las_path_init if las_path.endswith('.las') or las_path.endswith('.laz')]
        cdf = CloudDataFrame({'las_path': las_path_init})
        cdf.n_threads = n_jobs

        if get_bounding_boxes == True:
            cdf._build_polygons()

        return(cdf)

    def set_index(self, *args):
        return CloudDataFrame(super(CloudDataFrame, self).set_index(*args))

    def par_apply(self, func, column='las_path', buffer_distance=0, *args):
        """
        Apply a function to each item in `column`. Allows for parallelization using the n_jobs argument. This is \
        achieved via :class:`joblib.Parallel` and :func:`joblib.delayed`.

        :param func: The user defined function, must accept as an argument the value of each row of `column`.
        :param column: The column to apply on, will be the first argument to func.
        :param buffer_distance: The distance to buffer and aggregate each tile.
        :param *args: Further arguments to `func`
        """
        from joblib import Parallel, delayed

        if buffer_distance > 0:
            self._buffer(buffer_distance)
            for i, geom in enumerate(self["bounding_box"]):
                intersecting = self._get_intersecting(i)
                clip_geom = self['buffered_bounding_box'].iloc[i]
                parent_cloud = pyfor.cloud.Cloud(self["las_path"].iloc[i])
                for path in intersecting["las_path"]:
                    adjacent_cloud = pyfor.cloud.Cloud(path)
                    parent_cloud.data._append(adjacent_cloud.data)

        output = Parallel(n_jobs=self.n_threads)(delayed(func)(plot_path, *args) for plot_path in self[column])
        return output

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
        bboxes = self.par_apply(self._get_bounding_box, column='las_path')
        self["bounding_box"] = [Polygon(((bbox[0], bbox[2]), (bbox[1], bbox[2]),
                                           (bbox[1], bbox[3]), (bbox[0], bbox[3]))) for bbox in bboxes]
        self.set_geometry("bounding_box", inplace = True)

    def _get_intersecting(self, tile_index):
        """
        Gets the intersecting tiles for the given tile_index

        :param: The index of the tile within the CloudDataFrame
        :return: A CloudDataFrame of intersecting tiles
        """
        # TODO Seek more efficient solution...
        # FIXME this is probably a sloppy way
        intersect_bool = self.intersects(self["buffered_bounding_box"].iloc[tile_index])
        intersect_cdf = CloudDataFrame(self[intersect_bool])
        intersect_cdf.n_threads = self.n_threads
        return intersect_cdf

    def _buffer(self, distance, in_place = True):
        """
        Buffers the CloudDataFrame geometries.
        :return: A new CloudDataFrame with buffered geometries.
        """
        # TODO implement in_place
        # also, pretty sloppy, consider relegating to a function, like "copy" or something
        norm_geoms = self["bounding_box"].copy()
        buffered = super(CloudDataFrame, self).buffer(distance)
        cdf = CloudDataFrame(self)
        cdf["bounding_box"] = norm_geoms
        cdf["buffered_bounding_box"] = buffered
        cdf.n_threads = self.n_threads
        cdf.set_geometry("bounding_box", inplace=True)
        return cdf

    def plot(self, **kwargs):
        """
        Plots the bounding boxes of the Cloud objects.

        :param **kwargs: Keyword arguments to :meth:`geopandas.GeoDataFrame.plot`.
        """
        plot = super(CloudDataFrame, self).plot(**kwargs)
        plot.figure.show()

    @property
    def bounding_box(self):
        """Retrieves the bounding box for the entire collection. As a tuple (minx, muny, maxx, maxy)"""
        minx, miny, maxx, maxy = [i.bounds[0] for i in self['bounding_box']], [i.bounds[1] for i in self['bounding_box']], \
                                 [i.bounds[2] for i in self['bounding_box']], [i.bounds[3] for i in self['bounding_box']]
        col_bbox = np.min(minx), np.min(miny), np.max(maxx), np.max(maxy)
        return col_bbox

    def retile(self, out_dir, verbose = False):
        """
        Retiles the collection into `out_dir`. This is a simplified retiling function that splits each tile into \
        quadrants and writes these new quadrants to disk. The written files will be the original file name with an index
        appended. 0 is the bottom left quadrant, 1 is the bottom right, 2 is the top left and 3 is the top right.
        """
        from shapely.geometry import Polygon

        for index, row in self.iterrows():
            # Build the quadrant geometries, this is defined by the following six values
            larger_cell = row['bounding_box'].bounds

            x0, y0 = larger_cell[0], larger_cell[1]
            x2, y2 = larger_cell[2], larger_cell[3]
            x1 = ((larger_cell[2] - larger_cell[0]) / 2) + x0
            y1 = ((larger_cell[3] - larger_cell[1]) / 2) + y0

            # Create the geometries
            bottom_left = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
            bottom_right = Polygon([(x1, y0), (x1, y1), (x2, y1), (x2, y0)])
            top_left = Polygon([(x0, y1), (x0, y2), (x1, y2), (x1, y1)])
            top_right = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

            quadrants = [bottom_left, bottom_right, top_left, top_right]


            larger_cloud = pyfor.cloud.Cloud(row['las_path'])

            for i, quad in enumerate(quadrants):
                clipped = larger_cloud.clip(quad)
                clipped.write(os.path.join(out_dir, '{}_{}.las'.format(larger_cloud.name, i)))

    def _retile2(self, width, height, dir):
        """
        Retiles the collection and writes the new tiles to the directory defined in `dir`.

        :param width: The width of the new tiles.
        :param height: The height of the new tiles
        :param dir: The directory to write the new tiles.
        """
        # TODO Handle "edge" smaller tiles that straddle more than one larger tile
        from shapely.geometry import MultiLineString
        from shapely.ops import polygonize
        import warnings

        warnings.warn('_retile2 is in development, use at your own risk.', UserWarning)

        colbbox = self.bounding_box

        # TODO adding the constant makes it an inclusive stop, seems sloppy
        x = np.arange(colbbox[0], colbbox[2] + width, width)
        y = np.arange(colbbox[1], colbbox[3] + height, height)

        hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
        vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]

        grids = gpd.GeoSeries(polygonize(MultiLineString(hlines + vlines)))
        return(grids)

        ## Iterate through each original tile and find the intersecting new tile
        ## TODO better way for this?
        #for index, row in self.iterrows():
        #    # Find the set of smaller tiles that intersect with the larger
        #    # Load the larger cell into memory
        #    original_cell = pyfor.cloud.Cloud(row['las_path'])
        #    for i, new_cell in enumerate(grids[grids.intersects(row['bounding_box'])]):
        #        pc = original_cell.clip(new_cell)
        #        pc.write(os.path.join(dir, '{}_{}{}'.format(original_cell.name, i, original_cell.extension)))

    def create_index(self):
        """
        For each file in the collection, creates `.lax` files for spatial indexing using the default values.
        """
        for las_path in self['las_path']:
            laxpy.file.init_lax(las_path)

    def _index_las(self, las_path):
        """
        Checks if an equivalent `.lax` file exists. If so, creates a laxpy.IndexedLAS object, otherwise an error is thrown.
        """
        lax_path = las_path[:-1] + 'x'

        if os.path.isfile(lax_path):
            return laxpy.IndexedLAS(las_path)
        else:
            raise FileNotFoundError('There is no equivalent .lax file for this .las file.')

    def _merge_parents(self, parent_list, func, args):
        """
        Used in retiling and project level clipping operations.

        :return:
        """

        if len(parent_list) > 0:
            first = pyfor.cloud.Cloud(parent_list[0])

            for parent in parent_list[1:]:
                pc = pyfor.cloud.Cloud(parent)
                first.data._append(pc.data)

            func(first, *args)

    def _get_parents(self, polygon):
        return self[self['bounding_box'].intersects(polygon)]

    def _square_buffer(self, polygon, buffer):
        from shapely.geometry import Polygon
        minx, miny, maxx, maxy = polygon.bounds
        n_minx, n_miny, n_maxx, n_maxy = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer

        buffered_poly = Polygon([[n_minx, n_miny],
                                 [n_minx, n_maxy],
                                 [n_maxx, n_maxy],
                                 [n_maxx, n_miny]])
        return buffered_poly

    def _clip_no_index(self, polygons, func):
        # TODO clean this function, parallelize, etc.
        """
        A very rough way to bypass indexing for clipping, currently in development
        :param polygons:
        :return:
        """
        from joblib import Parallel, delayed

        # FIXME spatial join would be optimized
        parents = {}
        for i, poly in enumerate(polygons):
            poly_parents = []
            for ix, row in self.iterrows():
                if row['bounding_box'].intersects(poly):
                    poly_parents.append(row['las_path'])
                parents[i] = poly_parents

        Parallel(n_jobs=self.n_threads)(delayed(self._merge_parents)(parent_list, func, [polygons[poly_index], poly_index]) for poly_index, parent_list in parents.items())

    def _retiling_grid(self, target_cell_size, original_tile_size, buffer = 0):
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

        bottom, left = self.bounding_box[1], self.bounding_box[0]
        top, right = self.bounding_box[3], self.bounding_box[2]

        new_tile_size = np.ceil(original_tile_size / target_cell_size) * target_cell_size
        extension = new_tile_size - original_tile_size

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
                if len(self._get_parents(new_tile)) > 0:
                    new_tiles.append(new_tile)

        return new_tiles



    def _clip_no_index1(self, polygons, func):
        # TODO clean this function, parallelize, etc.
        """
        A very rough way to bypass indexing for clipping, currently in development
        :param polygons:
        :return:
        """

        # TODO make this block its own function
        intersected_tiles = {}
        for ix, row in self.iterrows():
            tile_bbox, las_path = row['bounding_box'], row['las_path']
            for poly in polygons:
                if tile_bbox.intersects(poly):
                    if las_path in intersected_tiles:
                        intersected_tiles[las_path].append(poly)
                    else:
                        intersected_tiles[las_path] = [poly]


        # TODO make this block its own function
        parents = {}
        for i, poly in enumerate(polygons):
            poly_parents = []
            for las_path, poly_list in intersected_tiles.items():
                if poly in poly_list:
                    poly_parents.append(las_path)
                parents[i] = poly_parents

        # For each polygon (i.e. each index in parents) construct the clipped point cloud.
        # maybe this could just be done in the chunk above?
        for poly_index, parent_list in parents.items():
            poly = polygons[poly_index]

            # Load first parent, append to this
            first = pyfor.cloud.Cloud(parent_list[0])
            first.normalize(3)

            for parent in parent_list[1:]:
                pc = pyfor.cloud.Cloud(parent)
                pc.normalize(3)
                first.data._append(pc.data)

            first = first.clip(poly)

            func(first)

            first.crs = pyproj.Proj(init='epsg:26910').srs
            p80 = first.grid(30).raster(lambda z_vec: np.percentile(z_vec, 80), "z")
            p80.write('/home/bryce/Documents/Dissertation/Chapter3/data/grids/p80_debug/{}_new.tif'.format(poly_index))

    def _write_clip(self, polygon, poly_index, out_path, parent_list):
        """
        Internal function that clips an indexed las and writes to file. This is used exclusively by `.clip` and is meant to be parallelized

        :return:
        """
        indexed_parents = [self._index_las(parent_path) for parent_path in parent_list]
        header = indexed_parents[0].header

        parent_points = pd.concat([pd.DataFrame.from_records(parent.query_polygon(polygon, scale=True)) \
                                   for parent in indexed_parents])

        pc = pyfor.cloud.Cloud(pyfor.cloud.LASData(parent_points, header))

        # TODO generalize to other filetypes
        pc.write(out_path+'.las')

    def clip(self, polygons, path, poly_names=None, verbose=False):
        """
        A collection-level clipping method. This function is meant for efficient querying across the study area using \
        a set of polygons. This method requires the presence of `.lax` files in the collection directory. To generate \
        these `.lax` files please use :meth:`.create_index` first. Each polygon will be clipped and written to the \
        specified `path`.

        :param polygons: Either a list of shapely polygons. If only one polygon is required wrap into a list before hand.
        :param path: The output path of the clip.
        :param poly_names: A list of polygon names to use when writing to file.
        """
        # TODO currently does not take advantage of multi-threading
        # TODO also a bit long, may be best to break up
        head, tail = os.path.split(path)

        # Which tiles do I need to make an index for?
        # It could  be the case that the input polys intersect with the same tile, but are checked out of order
        # Building this dict requires a bit of overhead, but is more memory efficient in the worst case
        from joblib import Parallel, delayed

        intersected_tiles = {}
        for ix, row in self.iterrows():
            tile_bbox, las_path = row['bounding_box'], row['las_path']
            for poly in polygons:
                if tile_bbox.intersects(poly):
                    if las_path in intersected_tiles:
                        intersected_tiles[las_path].append(poly)
                    else:
                        intersected_tiles[las_path] = [poly]

        # Which polygons have which parents?
        parents = {}
        for i, poly in enumerate(polygons):
            poly_parents = []
            for las_path, poly_list in intersected_tiles.items():
                if poly in poly_list:
                    poly_parents.append(las_path)
                parents[i] = poly_parents

        # For each polygon (i.e. each index in parents) construct the clipped point cloud.
        # maybe this could just be done in the chunk above?
        for poly_index, parent_list in parents.items():
            poly = polygons[poly_index]
            indexed_parents = [self._index_las(parent_path) for parent_path in parent_list]
            header = indexed_parents[0].header
            # TODO This is slow, but should be addressed upstream in laxpy, especially _scale_points


            parent_points = pd.concat([pd.DataFrame.from_records(parent.query_polygon(poly, scale=True)) for parent in indexed_parents])

            print('Clipping polygon {} of {}'.format(poly_index + 1, len(polygons)))
            pc = pyfor.cloud.Cloud(pyfor.cloud.LASData(parent_points, header))

            if poly_names is not None:
                out_path = head + os.path.sep + str(poly_names[poly_index]) + '.las'
            else:
                out_path = head + os.path.sep + str(poly_index) + '.las'

            print('Writing to {}'.format(out_path))
            pc.write(out_path)

        if poly_names is not None:
            Parallel(n_jobs=self.n_threads)(delayed(self._write_clip)(polygons[poly_index], poly_index, head + os.path.sep + str(poly_names[poly_index]), parent_list) \
                                            for poly_index, parent_list in parents.items())
        else:
            Parallel(n_jobs=self.n_threads)(delayed(self._write_clip)(polygons[poly_index], poly_index, head + os.path.sep + str(poly_index), parent_list) \
                                            for poly_index, parent_list in parents.items())


    def standard_metrics(self, heightbreak, index=None):
        """
        Retrieves a set of 29 standard metrics, including height percentiles and other summaries.

        :param index: An iterable of indices to set as the output dataframe index.
        :return: A pandas dataframe of standard metrics.
        """
        from pyfor.metrics import standard_metrics

        get_metrics = lambda las_path: standard_metrics(pyfor.cloud.Cloud(las_path).data.points, heightbreak=heightbreak)
        metrics = pd.concat(self.par_apply(get_metrics), sort=False)

        if index:
            metrics.index = index

        return metrics



def from_dir(las_dir, **kwargs):
    """
    Constructs a CloudDataFrame from a directory of las files.

    :param las_dir: The directory of las files.
    :return: A CloudDataFrame constructed from the directory of las files.
    """

    return CloudDataFrame._from_dir(las_dir, **kwargs)

