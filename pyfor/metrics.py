import numpy as np
import pandas as pd
import pyfor.rasterizer

all_pct = (1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99)


def summarize_return_num(return_nums):
    """
    Gets the number of returns by return number.

    :param return_nums: A :class:`numpy.ndarray` of the return number of each point.
    :return: A :class:`pandas.Series` of return number counts by return number.
    """
    numbers, counts = np.unique(return_nums, return_counts=True)
    return pd.Series(counts, index=numbers)


def summarize_percentiles(z, pct=all_pct):
    """
    :param z: A :class:`numpy.ndarray` of z values.
    """
    return (np.percentile(z, pct), pct)


def pct_above_heightbreak(grid, r=0, heightbreak="mean"):
    """
    Calculates the percentage of first returns above the mean. This needs its own function because it summarizes
    multiple columns of the point cloud, and is therefore more complex than typical summarizations
    (i.e. percentiles). This returns a `pyfor.rasterizer.Raster` object.

    :param grid: A `pyfor.rasterizer.Grid` object
    :param r: The return number to constrain to. Must be a positive integer. If r=0, all points will be considered
    (this is the default behavior).
    :param heightbreak: The height at which to summarize. If a number is given, this will be interpreted as the height
    at which points will be considered "above". If the string "mean" is given (this is the default), will use the mean
    height of that cell, for example, to construct the "pct_above_mean" metric.
    """

    points = grid.cloud.data.points

    if heightbreak == "mean":
        # Compute mean z in each cell
        is_above = points["z"] > grid.expand(grid.cell_values(np.mean, "z"))
    else:
        is_above = points["z"] > heightbreak

    if r > 0:
        is_r = points["return_num"] == r
        denominator = grid.cell_counts(is_r)
        numerator = grid.cell_counts(is_r & is_above)
    else:
        denominator = grid.cell_counts()
        numerator = grid.cell_counts(is_above)

    values = np.divide(
        numerator,
        denominator,
        out=np.full(grid.n_cells, np.nan),
        where=denominator > 0,
    )

    return pyfor.rasterizer.Raster(values.reshape(grid.m, grid.n), grid)


def grid_percentile(grid, percentile):
    """
    Calculates a percentile raster.
    :param percentile: The percentile (a number between 0 and 100) to compute.
    """
    return grid.percentile_raster("z", percentile)


def z_max(grid):
    """
    Calculates maximum z value.
    """

    return grid.raster(np.max, "z")


def z_min(grid):
    """
    Calculates minimum z value.
    """

    return grid.raster(np.min, "z")


def z_std(grid):
    """
    Calculates standard deviation of z value.
    """

    return grid.raster(np.std, "z")


def z_var(grid):
    """
    Calculates variance of z value.
    """

    return grid.raster(np.var, "z")


def z_mean(grid):
    """
    Calculates mean of z value.
    """

    return grid.raster(np.mean, "z")


def z_iqr(grid):
    """
    Calculates interquartile range of z value.
    """

    cells, percentiles = grid.cell_percentiles("z", [25, 75])
    array = np.full(grid.n_cells, np.nan)
    array[cells] = percentiles[1] - percentiles[0]
    return pyfor.rasterizer.Raster(array.reshape(grid.m, grid.n), grid)


def vol_cov(grid, r, heightbreak):
    """
    Calculates the volume covariate (percentage first returns above two meters times mean z)
    """

    pct_r_above_hb = pct_above_heightbreak(grid, r, heightbreak)
    mean_z = grid.raster(np.mean, "z")
    # Overwrite pct_r1_above_2m array (to save memory)
    pct_r_above_hb.array = pct_r_above_hb.array * mean_z.array

    return pct_r_above_hb


def z_mean_sq(grid):
    """
    Calculates the square of the mean z value.
    """

    rast = z_mean(grid)
    rast.array = rast.array**2
    return rast


def canopy_relief_ratio(grid, mean_z_arr, min_z_arr, max_z_arr):
    # Cells with a single return, or with no returns at all, leave a zero denominator and are NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        crr_arr = (mean_z_arr - min_z_arr) / (max_z_arr - min_z_arr)
    crr_rast = pyfor.rasterizer.Raster(crr_arr, grid)
    return crr_rast


def return_num(grid, num):
    """Compute the number of returns that match `num` for a grid object"""
    points = grid.cloud.data.points
    counts = grid.cell_counts(points["return_num"] == num).astype(float)
    counts[counts == 0] = np.nan

    return pyfor.rasterizer.Raster(counts.reshape(grid.m, grid.n), grid)


def all_returns(grid):
    return grid.raster("count", "z")


def total_returns(grid):
    return all_returns(grid)


def standard_metrics_grid(grid, heightbreak):
    metrics_dict = {}
    metrics_dict["max_z"] = z_max(grid)
    metrics_dict["min_z"] = z_min(grid)
    metrics_dict["mean_z"] = z_mean(grid)
    metrics_dict["stddev_z"] = z_std(grid)
    metrics_dict["var_z"] = z_var(grid)
    metrics_dict["canopy_relief_ratio"] = canopy_relief_ratio(
        grid,
        metrics_dict["mean_z"].array,
        metrics_dict["min_z"].array,
        metrics_dict["max_z"].array,
    )

    for pct in all_pct:
        metrics_dict["p_" + str(pct)] = grid_percentile(grid, pct)

    metrics_dict["pct_r_1_above_{}".format(heightbreak)] = pct_above_heightbreak(
        grid, 1, heightbreak
    )
    metrics_dict["pct_r_1_above_mean".format(heightbreak)] = pct_above_heightbreak(
        grid, 1, "mean"
    )
    metrics_dict["pct_all_above_{}".format(heightbreak)] = pct_above_heightbreak(
        grid, 0, heightbreak
    )
    metrics_dict["pct_all_above_mean".format(heightbreak)] = pct_above_heightbreak(
        grid, 0, "mean"
    )
    return metrics_dict


def standard_metrics_cloud(points, heightbreak):
    metrics = pd.DataFrame()

    # Some values used multiple times
    mean_z = np.mean(points["z"])

    metrics["total_returns"] = [len(points)]

    # Get number of returns by return number
    for i, num in enumerate(summarize_return_num(points["return_num"])):
        metrics["r_{}".format(i + 1)] = [num]

    metrics["max_z"] = [np.max(points["z"])]
    metrics["min_z"] = [np.min(points["z"])]
    metrics["mean_z"] = [mean_z]
    metrics["median_z"] = [np.median(points["z"])]
    metrics["stddev_z"] = [np.std(points["z"])]
    metrics["var_z"] = [np.var(points["z"])]

    for pct_z, pct in zip(*summarize_percentiles(points["z"])):
        metrics["p_{}".format(pct)] = [pct_z]

    # "Cover metrics"
    metrics["canopy_relief_ratio"] = (metrics["mean_z"] - metrics["min_z"]) / (
        metrics["max_z"] - metrics["min_z"]
    )
    metrics["pct_r_1_above_{}".format(heightbreak)] = (
        np.sum((points["return_num"] == 1) & (points["z"] > heightbreak))
        / metrics["r_1"]
    )
    metrics["pct_r_1_above_mean"] = (
        np.sum((points["return_num"] == 1) & (points["z"] > mean_z)) / metrics["r_1"]
    )
    metrics["pct_all_above_{}".format(heightbreak)] = (
        np.sum(points["z"] > heightbreak) / metrics["total_returns"]
    )
    metrics["pct_all_above_mean"] = (
        np.sum(points["z"] > mean_z) / metrics["total_returns"]
    )

    return metrics
