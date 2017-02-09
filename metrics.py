"""Holds metrics"""

import numpy as np

def mean_height(xyz):
    return np.mean(xyz[:,2])

def median_height(xyz):
    return np.median(xyz[:,2])

def mode_height(xyz):
    from scipy import stats
    return stats.mode(xyz[:,2])

def percentile_height(xyz, percentile):
    return np.percentile(xyz[:,2], percentile)

def percent_above(xyz, threshold):
    total_points  = len(xyz[:,2])
    above_points = (xyz[:,2]>threshold).sum()
    return float(above_points/total_points)