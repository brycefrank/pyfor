import json
import numpy as np
from numba import vectorize, bool_, float64

# These are the lower level clipping functions.

def square_clip(points, bounds):
    """
    Clips a square from a tuple describing the position of the square.

    :param las_xy: A N x 2 numpy array of x and y coordinates, x in
    column 0
    :param bounds: A tuple of length 4, min y and max y coordinates of the square.
    :return: A boolean mask, true is within the square
    """

    # Extact x y coordinates from cloud
    xy = points[["x", "y"]]

    # Create masks for each axis
    x_in = (xy["x"] >= bounds[0]) & (xy["x"] <= bounds[2])
    y_in = (xy["y"] >= bounds[1]) & (xy["y"] <= bounds[3])
    stack = np.stack((x_in, y_in), axis=1)
    in_clip = np.all(stack, axis=1)

    return(in_clip)

def ray_trace(x, y, poly):
    """
    A numba implementation of the ray tracing algorithm.

    :param x: A 1D numpy array of x coordinates.
    :param y: A 1D numpy array of y coordinates.
    :param poly: The coordinates of a polygon as a numpy array (i.e. from geo_json['coordinates']
    :return:
    """
    @vectorize([bool_(float64, float64)])
    def ray(x, y):
        # where xy is a coordinate
        n = len(poly)
        inside = False
        p2x = 0.0
        p2y = 0.0
        xints = 0.0
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    return(ray(x, y))


def poly_clip(points, poly):
    """
    Returns the indices within a given polygon.

    :param cloud: A cloud object.
    :param poly: A shapely Polygon, with coordinates in the same CRS as the point cloud (for best results).
    :return: A 1D numpy array of indices corresponding to points within the given polygon.
    """
    # Clip to bounding box
    bbox = poly.bounds
    pre_clip_mask = square_clip(points, bbox)
    pre_clip = points[["x", "y"]].iloc[pre_clip_mask].values

    # Store old indices
    pre_clip_inds = np.where(pre_clip_mask)[0]

    # Clip the preclip
    poly_coords = np.stack((poly.exterior.coords.xy[0],
                            poly.exterior.coords.xy[1]), axis = 1)

    full_clip_mask = ray_trace(pre_clip[:,0], pre_clip[:,1], poly_coords)
    clipped = pre_clip_inds[full_clip_mask]

    return(clipped)
