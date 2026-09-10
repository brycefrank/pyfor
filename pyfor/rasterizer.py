# Functions for rasterizing
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyfor import gisexport
import pyfor.metrics


def sample_array(array, bins_x, bins_y):
    """
    Looks up the value of a raster array for each point from its column and row bins.

    :param array: A 2D numpy array indexed as ``array[bins_y, bins_x]``.
    :param bins_x: A 1D integer array of the column bin of each point.
    :param bins_y: A 1D integer array of the row bin of each point.
    :return: A 1D float array of values, NaN where the bin of a point is outside of `array`.
    """
    values = np.full(len(bins_x), np.nan)
    inside = (
        (bins_x >= 0)
        & (bins_x < array.shape[1])
        & (bins_y >= 0)
        & (bins_y < array.shape[0])
    )
    values[inside] = array[bins_y[inside], bins_x[inside]]

    return values


def _sum_cells(values, cells, n_cells):
    return np.bincount(cells, weights=values, minlength=n_cells)


def _count_cells(values, cells, n_cells):
    return np.bincount(cells, minlength=n_cells)


def _mean_cells(values, cells, n_cells):
    counts = _count_cells(values, cells, n_cells)
    return np.divide(
        _sum_cells(values, cells, n_cells), counts, out=np.full(n_cells, np.nan), where=counts > 0
    )


def _extreme_cells(values, cells, n_cells, extreme):
    counts = _count_cells(values, cells, n_cells)
    empty = counts == 0
    result = np.full(n_cells, -np.inf if extreme is np.maximum else np.inf)
    extreme.at(result, cells, values)
    result[empty] = np.nan
    return result


def _min_cells(values, cells, n_cells):
    return _extreme_cells(values, cells, n_cells, np.minimum)


def _max_cells(values, cells, n_cells):
    return _extreme_cells(values, cells, n_cells, np.maximum)


def _var_cells(values, cells, n_cells):
    counts = _count_cells(values, cells, n_cells)
    mean = _mean_cells(values, cells, n_cells)
    # Two passes over the values keep the result stable for the height values pyfor deals with.
    deviations = values - mean[cells]
    return np.divide(
        _sum_cells(deviations * deviations, cells, n_cells),
        counts,
        out=np.full(n_cells, np.nan),
        where=counts > 0,
    )


def _std_cells(values, cells, n_cells):
    return np.sqrt(_var_cells(values, cells, n_cells))


# Reductions that can be computed for every cell at once, either by name or by numpy function. Any other
# callable is applied to the values of each occupied cell in turn.
_CELL_REDUCTIONS = {
    "count": _count_cells,
    "size": _count_cells,
    np.sum: _sum_cells,
    "sum": _sum_cells,
    np.mean: _mean_cells,
    "mean": _mean_cells,
    np.min: _min_cells,
    "min": _min_cells,
    np.max: _max_cells,
    "max": _max_cells,
    np.std: _std_cells,
    "std": _std_cells,
    np.var: _var_cells,
    "var": _var_cells,
}


def reduce_cells(values, cells, n_cells, func):
    """
    Reduces the values of every occupied cell of a grid with `func`.

    :param values: A 1D array of values, one per point.
    :param cells: A 1D integer array of the cell id of each point.
    :param n_cells: The total number of cells, occupied or not.
    :param func: Either the name of a reduction ("count", "sum", "mean", "min", "max", "std", "var"), a
     numpy function (i.e. :func:`np.max`), or any callable that reduces a 1D array of values to a scalar
     (i.e. :func:`np.percentile` or a lambda).
    :return: A tuple (cell_ids, results) of the occupied cells, ordered by ascending cell id.
    """
    reduction = _CELL_REDUCTIONS.get(func)
    if reduction is None:
        return _reduce_cells_generic(values, cells, func)

    counts = _count_cells(values, cells, n_cells)
    occupied = np.flatnonzero(counts > 0)
    return occupied, reduction(values, cells, n_cells)[occupied]


def _reduce_cells_generic(values, cells, func):
    """
    Applies an arbitrary callable to the values of each occupied cell. The values are ordered by cell once,
    so the callable is invoked with a contiguous view of the values of a cell.
    """
    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
    )
    bounds = np.concatenate((starts, [len(sorted_cells)]))
    sorted_values = values[order]

    results = [
        func(sorted_values[bounds[i] : bounds[i + 1]]) for i in range(len(starts))
    ]

    return sorted_cells[starts], np.asarray(results)


def percentile_cells(values, cells, percentiles):
    """
    Computes percentiles of the values of every occupied cell. This is a vectorized equivalent of calling
    :func:`numpy.percentile`, with its default linear interpolation, on the values of each cell, which is the
    most expensive reduction pyfor performs.

    :param values: A 1D array of values, one per point.
    :param cells: A 1D integer array of the cell id of each point.
    :param percentiles: A sequence of percentiles between 0 and 100.
    :return: A tuple (cell_ids, results) where results has shape ``(len(percentiles), len(cell_ids))``.
    """
    order = np.lexsort((values, cells))
    sorted_cells = cells[order]
    sorted_values = values[order].astype(np.float64, copy=False)

    starts = np.flatnonzero(
        np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
    )
    counts = np.diff(np.concatenate((starts, [len(sorted_cells)])))

    results = np.empty((len(percentiles), len(starts)))
    for i, percentile in enumerate(percentiles):
        positions = np.asarray(percentile, dtype=np.float64) / 100 * (counts - 1)
        lower = np.floor(positions).astype(np.int64)
        upper = np.ceil(positions).astype(np.int64)
        weight = positions - lower

        low = sorted_values[starts + lower]
        high = sorted_values[starts + upper]
        results[i] = low + (high - low) * weight

    return sorted_cells[starts], results


class GridSpec(NamedTuple):
    """
    The grid a raster is computed on: the bottom left corner, the cell size, and the size in cells.

    :param origin_x: The x coordinate of the left edge of the grid.
    :param origin_y: The y coordinate of the bottom edge of the grid.
    :param cell_size: The size of a cell, in the units of the point cloud.
    :param n: The number of cells in the x direction.
    :param m: The number of cells in the y direction.
    """

    origin_x: float
    origin_y: float
    cell_size: float
    n: int
    m: int

    @classmethod
    def covering(cls, min_x, min_y, max_x, max_y, cell_size):
        """
        The grid that covers an extent, with its origin snapped to a multiple of the cell size.

        Snapping is the target aligned pixels convention (`gdal_translate -tap`, and the default in
        terra and lidR). It is what makes rasters from neighbouring tiles line up: every snapped
        origin sits on the same lattice of multiples of ``cell_size``, so two tiles of a project
        describe the same cells, and a raster can be mosaicked without resampling. Tools whose grid
        defaults to the extent of the data, pyfor before this and PDAL's ``writers.gdal``, produce
        rasters that cannot be compared or mosaicked cell by cell.
        """
        origin_x = float(np.floor(min_x / cell_size) * cell_size)
        origin_y = float(np.floor(min_y / cell_size) * cell_size)

        return cls(
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=float(cell_size),
            n=max(1, int(np.ceil((max_x - origin_x) / cell_size))),
            m=max(1, int(np.ceil((max_y - origin_y) / cell_size))),
        )

    @property
    def bounds(self):
        """:return: A tuple (min_x, min_y, max_x, max_y) of the grid."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.n * self.cell_size,
            self.origin_y + self.m * self.cell_size,
        )

    @property
    def shape(self):
        """:return: The shape (m, n) of a raster on this grid."""
        return (self.m, self.n)

    @property
    def affine(self):
        """:return: The rasterio affine transformation of the grid, north up."""
        from rasterio.transform import from_origin

        return from_origin(
            self.origin_x,
            self.origin_y + self.m * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def covers(self, min_x, min_y, max_x, max_y):
        """:return: True if the grid covers the given extent."""
        left, bottom, right, top = self.bounds
        return min_x >= left and max_x <= right and min_y >= bottom and max_y <= top


def cell_bins(points, spec):
    """
    The column and row bin of every point on a grid.

    Cells are half open and counted from the bottom left corner, which is how GDAL and rasterio
    locate a point in a raster: a point at ``origin_y + k * cell_size`` is in row ``k``, so it is in
    the cell above the line when the raster is read north up. Bins are clipped into the grid, which
    only affects points lying exactly on the far edge of the extent; GDAL would consider those
    outside the raster and PDAL drops them.

    :param points: A structured numpy array of points, as held in :attr:`CloudData.points`.
    :param spec: A :class:`.GridSpec`.
    :return: A tuple (bins_x, bins_y) of 1D integer arrays, counted from the top left of the raster.
    """
    bins_x = np.floor((points["x"] - spec.origin_x) / spec.cell_size).astype(np.int64)
    bins_y = spec.m - 1 - np.floor((points["y"] - spec.origin_y) / spec.cell_size).astype(
        np.int64
    )

    np.clip(bins_x, 0, spec.n - 1, out=bins_x)
    np.clip(bins_y, 0, spec.m - 1, out=bins_y)
    return bins_x, bins_y


class Grid:
    """The Grid object is a representation of a point cloud that has been sorted into X and Y dimensional bins. From \
    the Grid object we can derive other useful products, most importantly, :class:`.Raster` objects.
    """

    def __init__(self, cloud, cell_size=None, spec=None):
        """
        Upon initialization, the parent cloud object's points are sorted into bins, which are held by the Grid \
        rather than by the cloud. Other useful information, such as the resolution, number of rows and columns are \
        also stored.

        :param cloud: The "parent" cloud object.
        :param cell_size: The size of the cell for sorting in the units of the input cloud object. Ignored if \
        `spec` is given, and it must agree with the cell size of `spec` if both are given.
        :param spec: An optional :class:`.GridSpec` to grid the cloud on. Passing the same spec for several tiles \
        is how rasters from those tiles are made to line up. By default the grid covers the cloud with its origin \
        snapped to a multiple of the cell size, the target aligned pixels convention.
        """

        self.cloud = cloud
        self._user_spec = spec

        min_x, max_x = self.cloud.data.min[0], self.cloud.data.max[0]
        min_y, max_y = self.cloud.data.min[1], self.cloud.data.max[1]

        if spec is None:
            if cell_size is None:
                raise ValueError("Either cell_size or spec must be given.")
            self.spec = GridSpec.covering(min_x, min_y, max_x, max_y, cell_size)
        else:
            if cell_size is not None and cell_size != spec.cell_size:
                raise ValueError(
                    "cell_size {} does not match the cell size {} of the given grid spec.".format(
                        cell_size, spec.cell_size
                    )
                )
            if not spec.covers(min_x, min_y, max_x, max_y):
                raise ValueError(
                    "The given grid spec covers {} but the cloud spans {}. A grid has to cover "
                    "every point it is asked to bin.".format(
                        spec.bounds, (min_x, min_y, max_x, max_y)
                    )
                )
            self.spec = spec

        self.cell_size = self.spec.cell_size
        self.m, self.n = self.spec.m, self.spec.n

        self.bins_x, self.bins_y = cell_bins(self.cloud.data.points, self.spec)
        self.cell_ids = self.bins_y * self.n + self.bins_x

    def _update(self):
        self.cloud.data._update()
        self.__init__(self.cloud, self.cell_size, self._user_spec)

    @property
    def n_cells(self):
        """:return: The total number of cells in the grid, occupied or not."""
        return self.m * self.n

    def reduce(self, func, dim, mask=None):
        """
        Reduces a dimension of the parent cloud for every occupied cell.

        :param func: A function to reduce the values of a cell, see :func:`.reduce_cells`.
        :param dim: The dimension (i.e. column name of the points) to reduce.
        :param mask: An optional boolean mask, only the selected points are reduced.
        :return: A tuple (bins_y, bins_x, values) of the occupied cells.
        """
        values = self.cloud.data.points[dim]
        cells = self.cell_ids
        if mask is not None:
            values, cells = values[mask], cells[mask]

        cells, results = reduce_cells(values, cells, self.n_cells, func)
        return cells // self.n, cells % self.n, results

    def cell_ranks(self):
        """
        The ordinal of each point within its own cell, in the order the points are held by the cloud.

        :return: A 1D integer array, zero for the first point of every cell.
        """
        order = np.argsort(self.cell_ids, kind="stable")
        sorted_cells = self.cell_ids[order]
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
        )
        group_start = np.repeat(starts, np.diff(np.concatenate((starts, [len(order)]))))

        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order)) - group_start
        return ranks

    def cell_counts(self, mask=None):
        """
        Counts the points of the parent cloud in every cell.

        :param mask: An optional boolean mask selecting the points to count.
        :return: A 1D integer array of length :attr:`n_cells`.
        """
        cells = self.cell_ids if mask is None else self.cell_ids[mask]
        return np.bincount(cells, minlength=self.n_cells)

    def cell_values(self, func, dim, mask=None):
        """
        Reduces a dimension of the parent cloud for every cell of the grid.

        :param func: A function to reduce the values of a cell, see :func:`.reduce_cells`.
        :param dim: The dimension (i.e. column name of the points) to reduce.
        :param mask: An optional boolean mask, only the selected points are reduced.
        :return: A 1D array of length :attr:`n_cells`, NaN where a cell is empty. The value of the cell
         (bins_y, bins_x) is at index ``bins_y * n + bins_x``.
        """
        bins_y, bins_x, values = self.reduce(func, dim, mask)
        array = np.full(self.n_cells, np.nan)
        array[bins_y * self.n + bins_x] = values
        return array

    def cell_percentiles(self, dim, percentiles):
        """
        Computes percentiles of a dimension of the parent cloud for every occupied cell.

        :param dim: The dimension (i.e. column name of the points) to reduce.
        :param percentiles: A sequence of percentiles between 0 and 100.
        :return: A tuple (cell_ids, results) where results has shape ``(len(percentiles), len(cell_ids))``.
        """
        return percentile_cells(
            self.cloud.data.points[dim], self.cell_ids, percentiles
        )

    def percentile_raster(self, dim, percentile):
        """
        A raster of a single percentile of a dimension of the parent cloud.

        :param dim: The dimension (i.e. column name of the points) to reduce.
        :param percentile: A percentile between 0 and 100.
        :return: A :class:`.Raster` object.
        """
        cells, results = self.cell_percentiles(dim, [percentile])
        array = np.full(self.n_cells, np.nan)
        array[cells] = results[0]
        return Raster(array.reshape(self.m, self.n), self)

    def expand(self, values):
        """
        Casts per cell values back onto the points of the parent cloud.

        :param values: A 1D array of length :attr:`n_cells`, i.e. the result of :meth:`.cell_values`.
        :return: A 1D array with the value of the cell of each point.
        """
        return values[self.cell_ids]

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
        return Raster(
            self.cell_values(func, dim, **kwargs).reshape(self.m, self.n), self
        )

    @property
    def empty_cells(self):
        """
        Retrieves the cells with no returns in self.data

        return: An N x 2 numpy array where each row cooresponds to the [y x] coordinate of the empty cell.
        """
        array = self.raster("count", "z").array
        emptys = np.argwhere(np.isnan(array))

        return emptys

    def interpolate(self, func, dim, interp_method="nearest", mask=None):
        """
        Interpolates missing cells in the grid. This function uses scipy.griddata as a backend. Please see \
        documentation for that function for more details.

        :param func: The function (or function string) to calculate an array on the gridded data.
        :param dim: The dimension (i.e. column name of self.cells) to cast func onto.
        :param interp_method: The interpolation method call for scipy.griddata, one of any: "nearest", "cubic", \
        "linear"
        :param mask: An optional boolean mask, only the selected points are interpolated.

        :return: An interpolated array.
        """
        from scipy.interpolate import griddata

        # Get the cells and values that we already have
        bins_y, bins_x, values = self.reduce(func, dim, mask)

        X, Y = np.mgrid[0 : self.n, 0 : self.m]

        # TODO generally a slow approach
        interp_grid = griddata(
            np.stack((bins_x, bins_y), axis=1), values, (X, Y), method=interp_method
        ).T

        return Raster(interp_grid, self)

    def metrics(self, func_dict, as_raster=False):
        """
        Calculates summary statistics for each grid cell in the Grid.

        :param func_dict: A dictionary containing keys corresponding to the columns of self.data and values that \
        correspond to the functions to be  called on those columns.
        :return: A pandas dataframe with the aggregated metrics.
        """

        # Reduce every requested metric. A dimension may be given a single function or a list of them,
        # as the previous groupby aggregation accepted.
        requests = []
        for dim, funcs in func_dict.items():
            if isinstance(funcs, (list, tuple, set)):
                requests.extend((dim, func) for func in funcs)
            else:
                requests.append((dim, funcs))

        aggregate = {}
        for dim, func in requests:
            aggregate[(dim, func)] = reduce_cells(
                self.cloud.data.points[dim], self.cell_ids, self.n_cells, func
            )

        if as_raster == False:
            # Assemble the reductions into a dataframe indexed by the cell bins, as the previous groupby
            # aggregation did
            index = pd.MultiIndex.from_arrays(
                [np.arange(self.n_cells) % self.n, np.arange(self.n_cells) // self.n],
                names=["bins_x", "bins_y"],
            )
            columns = {
                column: pd.Series(results, index=index[cell_ids]).reindex(index)
                for column, (cell_ids, results) in aggregate.items()
            }
            return pd.DataFrame(columns)

        rasters = []
        for (dim, func), (cell_ids, results) in aggregate.items():
            array = np.full((self.m, self.n), np.nan)
            array[cell_ids // self.n, cell_ids % self.n] = results
            rasters.append(Raster(array, self))

        # Get list of dimension names
        dims = [tup[0] for tup in aggregate.keys()]
        # Get list of metric names
        metrics = [tup[1] for tup in aggregate.keys()]
        return pd.DataFrame({"dim": dims, "metric": metrics, "raster": rasters}).set_index(
            ["dim", "metric"]
        )

    def standard_metrics(self, heightbreak=0):
        return pyfor.metrics.standard_metrics_grid(self, heightbreak=heightbreak)


class ImportedGrid(Grid):
    """
    ImportedGrid is used to normalize a parent cloud object with an arbitrary raster file. The grid is
    the grid of that raster, so the values of the raster can be looked up by the cell of a point.
    """

    def __init__(self, path, cloud):
        import rasterio

        with rasterio.open(path) as raster:
            if raster.transform.b != 0 or raster.transform.d != 0:
                raise ValueError(
                    "The input raster is rotated, which is not supported."
                )

            cell_size_x, cell_size_y = raster.transform[0], abs(raster.transform[4])
            if cell_size_x != cell_size_y:
                raise ValueError(
                    "Cell sizes not equal of input raster, not supported."
                )

            self.array = raster.read(1)
            self.bounds = raster.bounds
            self.crs = raster.crs

            self.spec = GridSpec(
                origin_x=raster.bounds.left,
                origin_y=raster.bounds.bottom,
                cell_size=cell_size_x,
                n=raster.width,
                m=raster.height,
            )

        self.cloud = cloud
        self._user_spec = self.spec
        self.cell_size = self.spec.cell_size
        self.m, self.n = self.spec.m, self.spec.n

        self.bins_x, self.bins_y = cell_bins(self.cloud.data.points, self.spec)
        self.cell_ids = self.bins_y * self.n + self.bins_x

    def _update(self):
        self.cloud.data._update()


class Raster:
    def __init__(self, array, grid):
        self.grid = grid
        self.cell_size = self.grid.cell_size
        self.array = array
        self._affine = self.grid.spec.affine

    def sample(self, bins_x, bins_y):
        """
        Looks up the value of the raster for each point from its column and row bins.

        :param bins_x: A 1D integer array of the column bin of each point.
        :param bins_y: A 1D integer array of the row bin of each point.
        :return: A 1D float array of values, NaN where the bin of a point is outside of the raster.
        """
        return sample_array(self.array, bins_x, bins_y)

    def force_extent(self, bbox):
        """
        Sets `self._affine` and `self.array` to a forced bounding box. Useful for trimming edges off of rasters when
        processing buffered tiles. This operation is done in place.

        The bounding box has to fall on the cells of the raster, otherwise the array cannot be trimmed to it
        without moving the data to different cells than the affine describes. A :class:`.ValueError` is raised
        rather than quietly rounding, which would leave an array labelled with a grid it is not on.

        :param bbox: Coordinates of output raster as a tuple (min_x, max_x, min_y, max_y)
        """
        from rasterio.transform import from_origin

        new_left, new_right, new_bot, new_top = bbox

        m, n = self.array.shape[0], self.array.shape[1]

        # Maniupulate the array to fit the new affine transformation
        old_left, old_top = self.grid.spec.origin_x, self.grid.spec.bounds[3]
        old_right, old_bot = (
            old_left + n * self.grid.cell_size,
            old_top - m * self.grid.cell_size,
        )

        diffs = (
            old_left - new_left,
            old_top - new_top,
            old_right - new_right,
            old_bot - new_bot,
        )
        cells = tuple(diff / self.cell_size for diff in diffs)
        if any(abs(cell - round(cell)) > 1e-6 for cell in cells):
            raise ValueError(
                "The bounding box {} does not fall on the cells of the raster, which start at "
                "({}, {}) with a cell size of {}. Trim by whole cells.".format(
                    bbox, old_left, old_bot, self.cell_size
                )
            )

        left_diff, top_diff, right_diff, bot_diff = (int(round(cell)) for cell in cells)

        if left_diff > 0:
            # bbox left is outside of raster left, we need to add columns of nans
            emptys = np.empty((m, left_diff))
            emptys[:] = np.nan
            self.array = np.insert(self.array, 0, np.transpose(emptys), axis=1)
        elif left_diff != 0:
            # bbox left is inside of raster left, we need to remove left diff columns
            self.array = self.array[:, abs(left_diff) :]

        if top_diff < 0:
            # bbox top is outside of raster top, we need to add rows of nans
            emptys = np.empty((abs(top_diff), self.array.shape[1]))
            emptys[:] = np.nan
            self.array = np.insert(self.array, 0, emptys, axis=0)
        elif top_diff != 0:
            # bbox top is inside of raster top, we need to remove rows of nans
            self.array = self.array[abs(top_diff) :, :]

        if right_diff < 0:
            # bbox right is outside of raster right, we need to add columns of nans
            emptys = np.empty((self.array.shape[0], abs(right_diff)))
            emptys[:] = np.nan
            self.array = np.append(self.array, emptys, axis=1)
        elif right_diff != 0:
            # bbox right is inside raster right, we need to remove columns
            self.array = self.array[:, :-right_diff]

        if bot_diff > 0:
            # bbox bottom is outside of raster bottom, we need to add rows of nans
            emptys = np.empty((abs(bot_diff), self.array.shape[1]))
            emptys[:] = np.nan
            self.array = np.append(self.array, emptys, axis=0)
        elif bot_diff != 0:
            # bbox bottom is inside of raster bottom, we need to remove columns
            self.array = self.array[:bot_diff, :]

        # Handle the affine transformation
        new_affine = from_origin(
            new_left, new_top, self.grid.cell_size, self.grid.cell_size
        )
        self._affine = new_affine

    def plot(self, cmap="viridis", block=False, return_plot=False):
        """
        Default plotting method for the Raster object.
        """

        # TODO implement cmap
        fig = plt.figure()
        ax = fig.add_subplot(111)
        caz = ax.matshow(self.array)
        fig.colorbar(caz)
        ax.xaxis.tick_bottom()
        ax.set_xticks(np.linspace(0, self.grid.n, 3))
        ax.set_yticks(np.flip(np.linspace(0, self.grid.m, 3)))

        x_ticks, y_ticks = (
            np.rint(
                np.linspace(self.grid.cloud.data.min[0], self.grid.cloud.data.max[0], 3)
            ),
            np.rint(
                np.linspace(self.grid.cloud.data.min[1], self.grid.cloud.data.max[1], 3)
            ),
        )

        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        if return_plot == True:
            return ax

        else:
            plt.show(block=block)

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

            warn(
                "No coordinate reference system defined. Please set the .crs attribute of the Cloud object.",
                UserWarning,
            )

        gisexport.array_to_raster(self.array, self._affine, self.grid.cloud.crs, path)
