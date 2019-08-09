import numpy as np
import pandas as pd
import pyfor.rasterizer

all_pct = (1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99)

def summarize_return_num(return_nums):
    """
    Gets the number of returns by return number.

    :param return_nums: A :class:`pandas.Series` of return number that describes the return number of each point.
    :return: A :class:`pandas.Series` of return number counts by return number.
    """
    return return_nums.groupby(return_nums).agg('count')

def summarize_percentiles(z, pct = all_pct):
    """
    :param z: A :class:`pandas.Series` of z values.
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

    if heightbreak == "mean":
        # Compute mean z in each cell
        mean_z = grid.cells.agg({'z': np.mean})
        mean_z = mean_z.rename(columns={'z': 'mean_z'})
        grid.cloud.data.points = pd.merge(grid.cloud.data.points, mean_z, on=['bins_x', 'bins_y'])
        is_above = grid.cloud.data.points['z'] > grid.cloud.data.points['mean_z']
    else:
        is_above = grid.cloud.data.points['z'] > heightbreak

    if r > 0:
        out_col = 'pct_r{}_above_{}'.format(r, heightbreak)
        grid.cloud.data.points['is_r'] = grid.cloud.data.points['return_num'] == r
        grid.cloud.data.points['is_r_above'] = grid.cloud.data.points['is_r'] & is_above
        cells = grid.cloud.grid(grid.cell_size).cells

        summary = cells.agg({'is_r': np.sum, 'is_r_above': np.sum}).reset_index()
        summary['bins_y'] = summary['bins_y']
        summary[out_col] = summary['is_r_above'] / summary['is_r']

    else:
        out_col = 'pct_all_above_{}'.format(heightbreak)
        grid.cloud.data.points['is_above'] = is_above
        cells = grid.cloud.grid(grid.cell_size).cells
        summary = cells.agg({'x': "count", 'is_above': np.sum}).reset_index()
        summary[out_col] = summary['is_above'] / summary['x']


    array = np.full((grid.m, grid.n), np.nan)
    array[summary["bins_y"], summary["bins_x"]] = summary[out_col]
    return pyfor.rasterizer.Raster(array, grid)


def grid_percentile(grid, percentile):
    """
    Calculates a percentile raster.
    :param percentile: The percentile (a number between 0 and 100) to compute.
    """
    return grid.raster(lambda z: np.percentile(z, percentile), "z")

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

    return grid.raster(lambda z: np.percentile(z, 75) - np.percentile(z, 25), "z")

def vol_cov(grid, r, heightbreak):
    """
    Calculates the volume covariate (percentage first returns above two meters times mean z)
    """

    pct_r_above_hb = pct_r_above_heightbreak(grid, r, heightbreak)
    mean_z = grid.raster(np.mean, "z")
    # Overwrite pct_r1_above_2m array (to save memory)
    pct_r_above_hb.array = pct_r_above_hb.array * mean_z.array

    return pct_r_above_hb

def z_mean_sq(grid):
    """
    Calculates the square of the mean z value.
    """

    rast = z_mean(grid)
    rast.array = rast.array^2
    return(rast)


def canopy_relief_ratio(grid, mean_z_arr, min_z_arr, max_z_arr):
    crr_arr = (mean_z_arr - min_z_arr) / (max_z_arr - min_z_arr)
    crr_rast = pyfor.rasterizer.Raster(crr_arr, grid)
    return(crr_rast)


def return_num(grid, num):
    """Compute the number of returns that match `num` for a grid object"""
    counts = grid.cells['return_num'].value_counts()
    counts = pd.DataFrame(counts)
    counts = counts.rename(columns = {'return_num': 'occurrences'})
    counts = counts.reset_index()
    counts = counts.loc[counts['return_num'] == num, :]

    array = np.full((grid.m, grid.n), np.nan)
    array[counts["bins_y"], counts["bins_x"]] = counts["occurrences"]

    return(pyfor.rasterizer.Raster(array, grid))

def total_returns(grid):
    counts = grid.cells['x'].count()
    counts = counts.reset_index()
    counts = counts.rename(columns = {'x': 'num'})

    array = np.full((grid.m, grid.n), np.nan)
    array[counts["bins_y"], counts["bins_x"]] = counts["num"]

    return(pyfor.rasterizer.Raster(array, grid))


def standard_metrics_grid(grid, heightbreak):
    # TODO all aboves
    metrics_dict = {}

    for pct in all_pct:
        metrics_dict['p_' + str(pct)] = grid_percentile(grid, pct)


    metrics_dict['max_z'] = z_max(grid)
    metrics_dict['min_z'] = z_min(grid)
    metrics_dict['mean_z'] = z_mean(grid)
    metrics_dict['stddev_z'] = z_std(grid)
    metrics_dict['var_z'] = z_var(grid)
    metrics_dict['canopy_relief_ratio'] = canopy_relief_ratio(grid, metrics_dict['mean_z'].array,
                                                         metrics_dict['min_z'].array, metrics_dict['max_z'].array)
    metrics_dict['pct_r_1_above_{}'.format(heightbreak)] = pct_r_above_heightbreak(grid, 1, heightbreak)
    metrics_dict['pct_r_1_above_mean'.format(heightbreak)] = pct_r_above_mean(grid, 1)
    return(metrics_dict)


def standard_metrics_cloud(points, heightbreak):
    metrics = pd.DataFrame()

    # Some values used multiple times
    mean_z = np.mean(points.z)

    metrics['total_returns'] = [np.alen(points)]

    # Get number of returns by return number
    for i, num in enumerate(summarize_return_num(points.return_num)):
        metrics['r_{}'.format(i+1)] = [num]

    metrics['max_z'] = [np.max(points.z)]
    metrics['min_z'] = [np.min(points.z)]
    metrics['mean_z'] = [mean_z]
    metrics['median_z'] = [np.median(points.z)]
    metrics['stddev_z'] = [np.std(points.z)]
    metrics['var_z'] = [np.var(points.z)]

    for pct_z, pct in zip(*summarize_percentiles(points.z)):
        metrics['p_{}'.format(pct)] = [pct_z]

    # "Cover metrics"
    metrics['canopy_relief_ratio'] = (metrics['mean_z'] - metrics['min_z']) / (metrics['max_z'] - metrics['min_z'])
    metrics['pct_r_1_above_{}'.format(heightbreak)] = np.sum((points['return_num'] == 1) & (points['z'] > heightbreak)) / metrics['r_1']
    metrics['pct_r_1_above_mean'] = np.sum((points['return_num'] == 1) & (points['z'] > mean_z)) / metrics['r_1']
    metrics['pct_all_above_{}'.format(heightbreak)] = np.sum(points['z'] > heightbreak) / metrics['total_returns']
    metrics['pct_all_above_mean'] = np.sum(points['z'] > mean_z) / metrics['total_returns']

    return metrics
