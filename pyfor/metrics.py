import numpy as np
import pandas as pd


def summarize_return_num(return_nums):
    """
    Gets the number of returns by return number.

    :param return_nums: A :class:`pandas.Series` of return number that describes the return number of each point.
    :return: A :class:`pandas.Series` of return number counts by return number.
    """

    return return_nums.groupby(return_nums).agg('count')


def make_percentile_funcs(percentiles):
    """
    Returns a dictionary of lambda percentile functions
    """
    return {'p_' + str(pct): lambda z: np.percentile(z, pct) for pct in percentiles}


def make_standard_funcs(percentiles, heightbreak, return_nums):
    """
    Returns a dictionary of standard metrics functions, such that each function takes as its first argument a vector of values.
    :param percentiles:
    :param heightbreak:
    :param return_nums: A vector of all possible return number values
    """

    # Percentiles
    standard_funcs = make_percentile_funcs(percentiles)

    # Standard z summaries
    standard_funcs['max_z'] = np.max
    standard_funcs['min_z'] = np.min
    standard_funcs['mean_z'] = np.mean
    standard_funcs['median_z'] = np.median
    standard_funcs['stddev_z'] = np.std
    standard_funcs['var_z'] = np.var

    # Return number summaries
    for i in return_nums:
        standard_funcs['r_{}'.format(i)] = lambda return_num_vec: np.sum(return_num_vec == i)
    standard_funcs['total_returns'] = lambda z: np.alen(z)

    # Canopy metrics
    standard_funcs['canopy_relief_ratio'] = lambda z: standard_funcs['min_z'](z) / (standard_funcs['max_z'](z) - standard_funcs['min_z'](z))
    standard_funcs['pct_all_above_{}'.format(heightbreak)] = lambda z: np.sum(z > heightbreak) / standard_funcs['total_returns'](z)
    standard_funcs['pct_r_1_above_{}'.format(heightbreak)] = lambda return_num_vec, z: np.sum((return_num_vec == 1) & (z > heightbreak)) / standard_funcs['r_1'](z)
    standard_funcs['pct_r_1_above_mean'] = lambda return_num_vec, z: np.sum((return_num_vec == 1) & (z > standard_funcs['mean_z'](z))) / standard_funcs['r_1'](z)
    standard_funcs['pct_all_above_{}'.format(heightbreak)] = lambda z: np.sum(z > heightbreak) / standard_funcs['total_returns'](z)
    standard_funcs['pct_all_above_mean'] = lambda z: np.sum(z > standard_funcs['mean_z'](z)) / standard_funcs['total_returns'](z)


    return standard_funcs

def summarize_percentiles(z, pct = (1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99)):
    """
    :param z: A :class:`pandas.Series` of z values.
    """

    return (np.percentile(z, pct), pct)


def standard_metrics(points, heightbreak=6):
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
