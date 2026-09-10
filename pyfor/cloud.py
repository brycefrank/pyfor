import copy

import laspy
import plyfile
import os
import pathlib
import warnings

import matplotlib
import numpy as np

from pyfor import clip
from pyfor import rasterizer

# Maps the column names used in CloudData.points to laspy dimension names. `x`, `y`, and `z`
# are laspy's scaled dimensions, the rest are the point format fields.
LAS_POINT_DIMS = {
    "x": "x",
    "y": "y",
    "z": "z",
    "intensity": "intensity",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "return_num": "return_number",
    "classification": "classification",
    "flag_byte": "bit_fields",
    "scan_angle_rank": "scan_angle_rank",
    "user_data": "user_data",
    "pt_src_id": "point_source_id",
}


def points_from_columns(columns):
    """
    Builds a structured numpy array of points from a mapping of column name to 1D array. This is the only
    place points are constructed, so every `CloudData.points` array has the same memory layout: one field
    per dimension, addressable by name.

    :param columns: A dictionary of column name to 1D numpy array.
    :return: A structured numpy array.
    """
    names = list(columns)
    if len(names) == 0:
        return np.empty(0, dtype=[])

    dtype = [(name, np.asarray(columns[name]).dtype) for name in names]
    points = np.empty(len(columns[names[0]]), dtype=dtype)
    for name in names:
        points[name] = columns[name]

    return points


def points_from_laspy(las, mask=None):
    """
    Extracts the pyfor point dimensions from a laspy object into a structured numpy array.

    :param las: A `laspy.LasData` or `laspy.ScaleAwarePointRecord` object.
    :param mask: An optional boolean mask or array of indices selecting a subset of the points.
    :return: A structured numpy array with one field per dimension present in the file.
    """
    columns = {}
    for column, dim in LAS_POINT_DIMS.items():
        if hasattr(las, dim):
            values = np.asarray(getattr(las, dim))
            columns[column] = values if mask is None else values[mask]

    return points_from_columns(columns)


def read_polygon(path, polygon, chunk_size=1000000):
    """
    Reads the points of a `.las` or `.laz` file that fall within a polygon without loading the whole
    file into memory. Points are read in chunks, pre-filtered by the polygon bounding box, and then
    tested against the polygon itself.

    :param path: The path of the `.las` or `.laz` file to read.
    :param polygon: A `shapely.geometry.Polygon` in the same CRS as the point cloud.
    :param chunk_size: The number of points to read per chunk.
    :return: A structured numpy array of the points within the polygon.
    """
    coords = np.stack(
        (polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]), axis=1
    )
    min_x, min_y, max_x, max_y = polygon.bounds

    chunks = []
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x, y = np.asarray(chunk.x), np.asarray(chunk.y)
            in_bounds = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            if not in_bounds.any():
                continue

            candidates = np.flatnonzero(in_bounds)
            inside = candidates[clip.ray_trace(x[in_bounds], y[in_bounds], coords)]
            if inside.size > 0:
                chunks.append(points_from_laspy(chunk, inside))

    if len(chunks) == 0:
        # An empty array still needs the fields of the file, so read a zero length record.
        with laspy.open(path) as reader:
            return points_from_laspy(reader.read_points(0))

    return np.concatenate(chunks)


# General class
class CloudData:
    def __init__(self, points, header):
        self.header = header
        self.points = points
        self._update()

    def _update(self):
        if len(self.points) == 0:
            # A cloud with no points has no extent, this is the state a clip that removed every
            # point leaves behind.
            self.min = [np.nan, np.nan, np.nan]
            self.max = [np.nan, np.nan, np.nan]
            self.count = 0
            return

        self.min = [
            np.min(self.points["x"]),
            np.min(self.points["y"]),
            np.min(self.points["z"]),
        ]
        self.max = [
            np.max(self.points["x"]),
            np.max(self.points["y"]),
            np.max(self.points["z"]),
        ]
        self.count = len(self.points)

    def _append(self, other):
        """
        Append one CloudData object to another.
        :return:
        """
        self.points = np.concatenate([self.points, other.points])
        self._update()


class PLYData(CloudData):
    def write(self, path):
        """
        Writes the object to file. This is a wrapper for :func:`plyfile.PlyData.write`

        :param path: The path of the ouput file.
        """
        if len(self.points) == 0:
            raise ValueError(
                "There is no data contained in this Cloud object, it is impossible to write."
            )

        elements = plyfile.PlyElement.describe(self.points, "vertex")
        plyfile.PlyData([elements]).write(path)


class LASData(CloudData):
    def write(self, path):
        """
        Writes the object to file. This is a wrapper for :meth:`laspy.LasData.write`, the header stored on the
        object is copied and its point count and bounds are updated to reflect the points held in memory.

        :param path: The path of the ouput file.
        """
        if len(self.points) == 0:
            raise ValueError(
                "There is no data contained in this Cloud object, it is impossible to write."
            )

        header = copy.deepcopy(self.header)
        header.point_count = len(self.points)
        las = laspy.LasData(header)
        for column, dim in LAS_POINT_DIMS.items():
            if column in self.points.dtype.names and hasattr(las, dim):
                setattr(las, dim, self.points[column])

        las.update_header()
        las.write(path)


class Cloud:
    """
    The cloud object is an API for interacting with `.las`, `.laz`, and `.ply` files in memory, and is generally \
    the starting point for any analysis with `pyfor`. For a more qualitative assessment of getting started with \
    :class:`Cloud` please see the \
    `user manual <https://github.com/brycefrank/pyfor_manual/blob/master/notebooks/2-ImportsExports.ipynb>`_.
    """

    def __init__(self, path):

        if isinstance(path, (str, pathlib.PurePath)):
            self.filepath = str(path)
            self.name = os.path.splitext(os.path.split(self.filepath)[1])[0]
            self.extension = os.path.splitext(self.filepath)[1]

            # A path to las or laz file
            if self.extension.lower() == ".las" or self.extension.lower() == ".laz":
                self._get_las_points(laspy.read(self.filepath))

            elif self.extension.lower() == ".ply":
                ply = plyfile.PlyData.read(path)
                ply_points = ply.elements[0].data
                points = points_from_columns(
                    {"x": ply_points["x"], "y": ply_points["y"], "z": ply_points["z"]}
                )
                header = "ply_header"
                self.data = PLYData(points, header)

            else:
                raise ValueError(
                    "File extension not supported, please input either a las, laz, ply or CloudData object."
                )

        elif isinstance(path, CloudData):
            self.data = path

            if isinstance(self.data.header, laspy.LasHeader):
                self.data = LASData(self.data.points, self.data.header)

            elif self.data.header == "ply_header":
                self.data = PLYData(self.data.points, self.data.header)

        elif isinstance(path, laspy.LasData):
            self._get_las_points(path)

        else:
            raise ValueError(
                "Object type not supported, please input either a file path with a supported extension or a CloudData object."
            )

        # We're not sure if this is true or false yet
        self.normalized = None

        # A coordinate reference system is only known if the file carries one.
        self.crs = None
        if isinstance(self.data.header, laspy.LasHeader):
            self.crs = self.data.header.parse_crs()

    @classmethod
    def from_pdal(cls, ins):
        """
        Converts a PDAL `ins` argument from a PDAL `filters.python` into a `Cloud` object.

        :param ins: The `ins` argument from PDAL.
        """
        rename = {"X": "x", "Y": "y", "Z": "z", "ReturnNumber": "return_num"}
        columns = {
            rename.get(name, name): np.asarray(values) for name, values in ins.items()
        }
        cloud_data = CloudData(points_from_columns(columns), header=None)
        return cls(cloud_data)

    def _get_las_points(self, las):
        """
        Reads points into a structured numpy array.

        :param las: A `laspy.LasData` object.
        """
        self.data = LASData(points_from_laspy(las), las.header)

    def __str__(self):
        """
        Returns a human readable summary of the Cloud object.
        """
        from os.path import getsize

        summary = {}
        summary["Minimum (x y z)"] = [
            float("{0:.2f}".format(elem)) for elem in self.data.min
        ]
        summary["Maximum (x y z)"] = [
            float("{0:.2f}".format(elem)) for elem in self.data.max
        ]
        summary["Number of Points"] = len(self.data.points)
        if hasattr(self, "extension"):
            summary["File Size"] = getsize(self.filepath)

            if self.extension.lower() == ".las" or self.extension.lower() == ".laz":
                summary["LAS Specification"] = self.data.header.version

        if self.crs is not None:
            summary["CRS"] = self.crs

        string_list = [key + ": " + str(val) + "\n" for key, val in summary.items()]
        return "".join(str(x) for x in string_list)

    def grid(self, cell_size, spec=None):
        """
        Generates a :class:`.Grid` object for the parent object given a cell size. \
        See the documentation for :class:`.Grid` for more information.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :param spec: An optional :class:`.GridSpec`. By default the grid covers the cloud and its origin is snapped \
        to a multiple of the cell size (the target aligned pixels convention), which is what makes rasters from \
        different tiles line up with each other.
        :return: A :class:`.Grid` object.
        """
        return rasterizer.Grid(self, cell_size, spec=spec)

    def plot(self, cell_size=1, cmap="viridis", return_plot=False, block=False):
        """
        Plots a basic canopy height model of the Cloud object. This is mainly a convenience function for \
        :class:`.Raster.plot`. More robust methods exist for dealing with canopy height models. Please see the \
        `user manual <https://github.com/brycefrank/pyfor_manual/blob/master/notebooks/3-CanopyHeightModel.ipynb>`_.

        :param clip_size: The resolution of the plot in the same units as the input file.
        :param return_plot: If true, returns a matplotlib plt object.
        :return: If return_plot == True, returns matplotlib plt object. Not yet implemented.
        """
        rasterizer.Grid(self, cell_size).raster("max", "z").plot(
            cmap, block=block, return_plot=return_plot
        )

    def plot3d(
        self,
        dim="z",
        point_size=1,
        cmap="Spectral_r",
        max_points=5e5,
        n_bin=8,
        plot_trees=False,
    ):
        """
        Plots the three dimensional point cloud using a `Qt` backend. By default, if the point cloud exceeds 5e5 \
         points, then it is downsampled using a uniform random distribution. This is for performance purposes.

        :param point_size: The size of the rendered points.
        :param dim: The dimension upon which to color (i.e. "z", "intensity", etc.)
        :param cmap: The matplotlib color map used to color the height distribution.
        :param max_points: The maximum number of points to render.
        """
        from pyqtgraph.Qt import QtCore, QtGui
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl

        if self.data.count > max_points:
            sample_mask = np.random.randint(self.data.count, size=int(max_points))
            coordinates = np.stack(
                [
                    self.data.points["x"],
                    self.data.points["y"],
                    self.data.points["z"],
                ],
                axis=1,
            )[sample_mask, :]

            color_dim = np.copy(self.data.points[dim][sample_mask])
            print("Too many points, down sampling for 3d plot performance.")
        else:
            coordinates = np.stack(
                [
                    self.data.points["x"],
                    self.data.points["y"],
                    self.data.points["z"],
                ],
                axis=1,
            )
            color_dim = np.copy(self.data.points[dim])

        # If dim is user data (probably TREE ID or some such thing) then we want a discrete colormap
        color_dim = (color_dim - np.min(color_dim)) / (
            np.max(color_dim) - np.min(color_dim)
        )
        cmap = matplotlib.colormaps[cmap]
        colors = cmap(color_dim)

        # Start Qt app and widget
        pg.mkQApp()
        view = gl.GLViewWidget()

        # Create the points, change to opaque, set size to 1
        points = gl.GLScatterPlotItem(pos=coordinates, color=colors)
        points.setGLOptions("opaque")
        points.setData(size=np.repeat(point_size, len(coordinates)))

        # Add points to the viewer
        view.addItem(points)

        # Center on the arithmetic mean of the point cloud and display
        center = np.mean(coordinates, axis=0)
        view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        # Very ad-hoc
        view.opts["distance"] = (self.data.max[0] - self.data.min[0]) * 1.2
        # return(view.opts)
        view.show()

    def normalize(self, cell_size, classified=False, spec=None, **kwargs):
        """
        Normalize the cloud using the default Zhang et al. (2003) progressive morphological ground filter. Please see \
        the documentation in :class:`.ground_filter.Zhang2003` for more information and keyword argument definitions. \
        If you want to use a pre-computed DEM to normalize, please see :meth:`.subtract`.

        :param cell_size: The resolution of the intermediate bare earth model.
        :param classified: If True and file type is `.las` or `.laz`, uses the points classified as ground (i.e. 2) to \
        construct the intermediate bare earth model.
        :param spec: An optional :class:`.GridSpec` for the bare earth model. Passing the same spec for every tile of \
        a project keeps the normalization of those tiles consistent with each other.
        """

        from pyfor.ground_filter import Zhang2003

        filter = Zhang2003(cell_size)
        filter.normalize(self, classified=classified, spec=spec)

    def subtract(self, path):
        """
        Normalize using a pre-computed raster file, i.e. "subtract" the heights from the input raster **in place**. \
        This assumes the raster and the point cloud are in the same coordinate system.
        :param path: The path to the raster file, must be in a format supported by `rasterio`.
        :return:
        """

        imported_grid = rasterizer.ImportedGrid(path, self)
        self.data.points["z"] = self.data.points["z"] - rasterizer.sample_array(
            imported_grid.array, imported_grid.bins_x, imported_grid.bins_y
        )
        self.data._update()

    def clip(self, polygon):
        """
        Clips the point cloud to the provided shapely polygon using a ray casting algorithm. This method calls \
        :func:`.clip.poly_clip` directly. This returns a new :class:`.Cloud`.

        :param polygon: A :class:`shapely.geometry.Polygon` in the same CRS as the Cloud.
        :return: A new :class:.`Cloud` object clipped to the provided polygon.
        """

        keep = clip.poly_clip(self.data.points, polygon)

        new_cloud = Cloud(CloudData(self.data.points[keep], self.data.header))

        # Warn user if the resulting cloud has no points.
        if len(new_cloud.data.points) == 0:
            warnings.warn("The clipped point cloud has no remaining points")

        return new_cloud

    def filter(self, min, max, dim):
        """
        Filters a cloud object for a given dimension **in place**.

        :param min: Minimum dimension to retain.
        :param max: Maximum dimension to retain.
        :param dim: The dimension of interest as a string. For example "z". This corresponds to a column label in \
        :attr:`self.data.points`.
        """
        condition = (self.data.points[dim] > min) & (self.data.points[dim] < max)
        self.data.points = self.data.points[condition]
        self.data._update()

    def chm(self, cell_size, interp_method=None, pit_filter=None, kernel_size=3, spec=None):
        """
        Returns a :class:`.Raster` object of the maximum z value in each cell, with optional interpolation \
         (i.e. nan-filling) and pit filter parameters. Currently, only a median pit filter is implemented.

        :param cell_size: The cell size for the returned raster in the same units as the parent Cloud or las file.
        :param interp_method: The interpolation method as a string to fill in NA values of the produced canopy height \
         model, one of either "nearest", "cubic", or "linear". This is an argument to `scipy.interpolate.griddata`.
        :param pit_filter: If "median" passes a median filter over the produced canopy height model.
        :param kernel_size: The kernel size of the median filter, must be an odd integer.
        :param spec: An optional :class:`.GridSpec` for the canopy height model, see :meth:`.grid`.
        :return: A :class:`.Raster` object of the canopy height model.
        """

        # TODO make user pass the function itself?
        if pit_filter == "median":
            raster = self.grid(cell_size, spec=spec).interpolate(
                "max", "z", interp_method=interp_method
            )
            raster.pit_filter(kernel_size=kernel_size)
            return raster

        if interp_method == None:
            return self.grid(cell_size, spec=spec).raster("max", "z")

        else:
            return self.grid(cell_size, spec=spec).interpolate(
                "max", "z", interp_method
            )

    def standard_metrics(self, heightbreak=0):
        from pyfor.metrics import standard_metrics_cloud

        return standard_metrics_cloud(self.data.points, heightbreak)

    @property
    def convex_hull(self):
        """
        Calculates the convex hull of the cloud projected onto a 2d plane, a wrapper for \
         :func:`scipy.spatial.ConvexHull`.

        :return: A :class:`shapely.geometry.Polygon` of the convex hull.
        """
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon

        hull = ConvexHull(
            np.stack((self.data.points["x"], self.data.points["y"]), axis=1)
        )

        return Polygon(hull.points[hull.vertices])

    def write(self, path):
        """
        Write to file. The precise mechanisms of this writing will depend on the file input type. For `.las` files \
        this will be handled by :meth:`.LASData.write`, for `.ply` files this will be handled by :meth:`.PLYData.write`.

        :param path: The path of the output file.
        """
        self.data.write(path)
