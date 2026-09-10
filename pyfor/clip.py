import numpy as np

# These are the lower level clipping functions.


def square_clip(points, bounds):
    """
    Clips a square from a tuple describing the position of the square.

    :param points: A structured numpy array of points, as held in :attr:`CloudData.points`.
    :param bounds: A tuple of length 4, min y and max y coordinates of the square.
    :return: A boolean mask, true is within the square, false is outside of the square.
    """

    # Create masks for each axis
    x_in = (points["x"] >= bounds[0]) & (points["x"] <= bounds[2])
    y_in = (points["y"] >= bounds[1]) & (points["y"] <= bounds[3])

    return x_in & y_in


def ray_trace(x, y, poly):
    """
    Determines for some set of x and y coordinates, which of those coordinates is within `poly`. Ray trace is \
    generally called as an internal function, see :func:`.poly_clip`

    :param x: A 1D numpy array of x coordinates.
    :param y: A 1D numpy array of y coordinates.
    :param poly: The coordinates of a polygon as a numpy array (i.e. from geo_json['coordinates']
    :return: A 1D boolean numpy array, true values are those points that are within `poly`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)

    # Each edge of the polygon (including the closing edge) either crosses the ray cast from a point or
    # it does not. A point is inside the polygon if an odd number of edges cross, so count the crossings
    # per edge and check the parity once every edge has been visited.
    p1x, p1y = np.roll(poly[:, 0], 1), np.roll(poly[:, 1], 1)
    p2x, p2y = poly[:, 0], poly[:, 1]

    # A horizontal edge can never satisfy this condition, so dividing by p2y - p1y below is safe where
    # the result is used.
    crossings = np.zeros(x.shape, dtype=np.int64)
    for i in range(len(poly)):
        active = np.flatnonzero(
            (y > min(p1y[i], p2y[i]))
            & (y <= max(p1y[i], p2y[i]))
            & (x <= max(p1x[i], p2x[i]))
        )
        if active.size == 0:
            continue

        # Only the active points reach the intersection test, and a horizontal edge is never active,
        # so the division here is always defined.
        active_y = y[active]
        xints = (
            (active_y - p1y[i]) * (p2x[i] - p1x[i]) / (p2y[i] - p1y[i]) + p1x[i]
        )
        crossings[active[(p1x[i] == p2x[i]) | (x[active] <= xints)]] += 1

    return (crossings % 2) == 1


def poly_clip(points, poly):
    """
    Returns the indices of `points` that are within a given polygon. This differs from :func:`.ray_trace` \
    in that it enforces a small "pre-clip" optimization by first clipping to the polygon bounding box. This function
    is directly called by :meth:`.Cloud.clip`.

    :param points: A structured numpy array of points, as held in :attr:`CloudData.points`.
    :param poly: A shapely Polygon, with coordinates in the same CRS as the point cloud.
    :return: A 1D numpy array of indices corresponding to points within the given polygon.
    """
    # Clip to bounding box
    bbox = poly.bounds
    pre_clip_mask = square_clip(points, bbox)

    # Store old indices
    pre_clip_inds = np.flatnonzero(pre_clip_mask)

    # Clip the preclip
    poly_coords = np.stack(
        (poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]), axis=1
    )

    full_clip_mask = ray_trace(
        points["x"][pre_clip_mask], points["y"][pre_clip_mask], poly_coords
    )

    return pre_clip_inds[full_clip_mask]
