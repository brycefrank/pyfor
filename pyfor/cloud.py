import laspy
import plyfile
import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from pyfor import rasterizer
from pyfor import clip
import pathlib
import warnings

# General class
class CloudData:
    def __init__(self, points, header):
        self.header = header
        self.points = points
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
        self.count = np.alen(self.points)

    def _update(self):
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
        self.count = np.alen(self.points)

    def _append(self, other):
        """
        Append one CloudData object to another.
        :return:
        """
        self.points = pd.concat([self.points, other.points], sort=False)
        self._update()


class PLYData(CloudData):
    def write(self, path):
        """
        Writes the object to file. This is a wrapper for :func:`plyfile.PlyData.write`

        :param path: The path of the ouput file.
        """
        if len(self.points) > 0:
            # coordinate_array = self.points[["x", "y", "z"]].values.T
            # vertex_array = list(zip(coordinate_array[0],coordinate_array[1], coordinate_array[2]))
            # vertex_array = np.array(vertex_array, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            vertex_array = self.points.to_records(index=False)
            elements = plyfile.PlyElement.describe(vertex_array, "vertex")
            plyfile.PlyData([elements]).write(path)
        else:
            raise ValueError(
                "There is no data contained in this Cloud object, it is impossible to write."
            )


class LASData(CloudData):
    def write(self, path):
        """
        Writes the object to file. This is a wrapper for :func:`laspy.file.File`

        :param path: The path of the ouput file.
        """
        if len(self.points) > 0:
            writer = laspy.file.File(path, header=self.header, mode="w")

            for dim in self.points:
                setattr(writer, dim, self.points[dim])

            writer.close()
        else:
            raise ValueError(
                "There is no data contained in this Cloud object, it is impossible to write."
            )


class Cloud:
    """
    The cloud object is an API for interacting with `.las`, `.laz`, and `.ply` files in memory, and is generally \
    the starting point for any analysis with `pyfor`. For a more qualitative assessment of getting started with \
    :class:`Cloud` please see the \
    `user manual <https://github.com/brycefrank/pyfor_manual/blob/master/notebooks/2-ImportsExports.ipynb>`_.
    """

    def __init__(self, path):

        if type(path) == str or type(path) == pathlib.PosixPath:
            self.filepath = path
            self.name = os.path.splitext(os.path.split(path)[1])[0]
            self.extension = os.path.splitext(path)[1]

            # A path to las or laz file
            if self.extension.lower() == ".las" or self.extension.lower() == ".laz":
                las = laspy.file.File(self.filepath)
                self._get_las_points(las)

            elif self.extension.lower() == ".ply":
                ply = plyfile.PlyData.read(path)
                ply_points = ply.elements[0].data
                points = pd.DataFrame(
                    {"x": ply_points["x"], "y": ply_points["y"], "z": ply_points["z"]}
                )
                header = "ply_header"
                self.data = PLYData(points, header)

            else:
                raise ValueError(
                    "File extension not supported, please input either a las, laz, ply or CloudData object."
                )

        elif type(path) == CloudData or isinstance(path, CloudData):
            self.data = path

            if type(self.data.header) == laspy.header.HeaderManager:
                self.data = LASData(self.data.points, self.data.header)

            elif self.data.header == "ply_header":
                self.data = PLYData(self.data.points, self.data.header)

        elif (
            path.__class__.__bases__[0] == laspy.file.File
            or type(path) == laspy.file.File
        ):
            self._get_las_points(path)

        else:
            raise ValueError(
                "Object type not supported, please input either a file path with a supported extension or a CloudData object."
            )

        # We're not sure if this is true or false yet
        self.normalized = None
        self.crs = None

    @classmethod
    def from_pdal(cls, ins):
        """
        Converts a PDAL `ins` argument from a PDAL `filters.python` into a `Cloud` object.

        :param ins: The `ins` argument from PDAL.
        """

        df = pd.DataFrame(ins)
        df = df.rename(
            columns={"X": "x", "Y": "y", "Z": "z", "ReturnNumber": "return_num"}
        )
        cloud_data = CloudData(df, header=None)
        return cls(cloud_data)

    def _get_las_points(self, las):
        """
        Reads points into pandas dataframe.

        :param las: A laspy.file.File (or subclass) object.
        """

        # Iterate over point format specification
        dims = [
            "x",
            "y",
            "z",
            "intensity",
            "red",
            "green",
            "blue",
            "return_num",
            "classification",
            "flag_byte",
            "scan_angle_rank",
            "user_data",
            "pt_src_id",
        ]

        points = {}
        for dim in dims:
            try:
                points[dim] = eval("las.{}".format(dim))
            except:
                pass
        points = pd.DataFrame(points)

        header = las.header
        self.data = LASData(points, header)

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

    def grid(self, cell_size):
        """
        Generates a :class:`.Grid` object for the parent object given a cell size. \
        See the documentation for :class:`.Grid` for more information.

        :param cell_size: The resolution of the plot in the same units as the input file.
        :return: A :class:`.Grid` object.
        """
        return rasterizer.Grid(self, cell_size)

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
                [self.data.points.x, self.data.points.y, self.data.points.z], axis=1
            )[sample_mask, :]

            color_dim = np.copy(self.data.points[dim].iloc[sample_mask].values)
            print("Too many points, down sampling for 3d plot performance.")
        else:
            coordinates = np.stack(
                [self.data.points.x, self.data.points.y, self.data.points.z], axis=1
            )
            color_dim = np.copy(self.data.points[dim].values)

        # If dim is user data (probably TREE ID or some such thing) then we want a discrete colormap
        color_dim = (color_dim - np.min(color_dim)) / (
            np.max(color_dim) - np.min(color_dim)
        )
        cmap = cm.get_cmap(cmap)
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

    def normalize(self, cell_size, classified=False, **kwargs):
        """
        Normalize the cloud using the default Zhang et al. (2003) progressive morphological ground filter. Please see \
        the documentation in :class:`.ground_filter.Zhang2003` for more information and keyword argument definitions. \
        If you want to use a pre-computed DEM to normalize, please see :meth:`.subtract`.

        :param cell_size: The resolution of the intermediate bare earth model.
        :param classified: If True and file type is `.las` or `.laz`, uses the points classified as ground (i.e. 2) to \
        construct the intermediate bare earth model.
        """

        from pyfor.ground_filter import Zhang2003

        filter = Zhang2003(cell_size)

        if classified:
            filter.bem(self, classified=classified)
            filter.normalize(self)
        else:
            filter.normalize(self)

    def subtract(self, path):
        """
        Normalize using a pre-computed raster file, i.e. "subtract" the heights from the input raster **in place**. \
        This assumes the raster and the point cloud are in the same coordinate system.
        :param path: The path to the raster file, must be in a format supported by `rasterio`.
        :return:
        """

        imported_grid = rasterizer.ImportedGrid(path, self)
        df = (
            pd.DataFrame(np.flipud(imported_grid.in_raster.read(1)))
            .stack()
            .rename_axis(["bins_y", "bins_x"])
            .reset_index(name="val")
        )
        df = self.data.points.reset_index().merge(df, how="left").set_index("index")
        self.data.points["z"] = df["z"] - df["val"]

    def clip(self, polygon):
        """
        Clips the point cloud to the provided shapely polygon using a ray casting algorithm. This method calls \
        :func:`.clip.poly_clip` directly. This returns a new :class:`.Cloud`.

        :param polygon: A :class:`shapely.geometry.Polygon` in the same CRS as the Cloud.
        :return: A new :class:.`Cloud` object clipped to the provided polygon.
        """

        keep = clip.poly_clip(self.data.points, polygon)

        # Create copy to avoid warnings
        keep_points = self.data.points.iloc[keep].copy()
        new_cloud = Cloud(CloudData(keep_points, self.data.header))
        new_cloud.data.points = new_cloud.data.points.reset_index(drop=True)
        new_cloud.data._update()

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

    def chm(self, cell_size, interp_method=None, pit_filter=None, kernel_size=3):
        """
        Returns a :class:`.Raster` object of the maximum z value in each cell, with optional interpolation \
         (i.e. nan-filling) and pit filter parameters. Currently, only a median pit filter is implemented.

        :param cell_size: The cell size for the returned raster in the same units as the parent Cloud or las file.
        :param interp_method: The interpolation method as a string to fill in NA values of the produced canopy height \
         model, one of either "nearest", "cubic", or "linear". This is an argument to `scipy.interpolate.griddata`.
        :param pit_filter: If "median" passes a median filter over the produced canopy height model.
        :param kernel_size: The kernel size of the median filter, must be an odd integer.
        :return: A :class:`.Raster` object of the canopy height model.
        """

        # TODO make user pass the function itself?
        if pit_filter == "median":
            raster = self.grid(cell_size).interpolate(
                "max", "z", interp_method=interp_method
            )
            raster.pit_filter(kernel_size=kernel_size)
            return raster

        if interp_method == None:
            return self.grid(cell_size).raster("max", "z")

        else:
            return self.grid(cell_size).interpolate("max", "z", interp_method)

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

        hull = ConvexHull(self.data.points[["x", "y"]].values)

        return Polygon(hull.points[hull.vertices])

    def write(self, path):
        """
        Write to file. The precise mechanisms of this writing will depend on the file input type. For `.las` files \
        this will be handled by :meth:`.LASData.write`, for `.ply` files this will be handled by :meth:`.PLYData.write`.

        :param path: The path of the output file.
        """
        self.data.write(path)
