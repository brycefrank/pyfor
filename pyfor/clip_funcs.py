import json
import numpy as np
from numba import vectorize, bool_, float64

def square_clip(cloud, bounds):
    """
    Clips a square from a tuple describing the position of the square

    :param las_xy: A N x 2 numpy array of x and y coordinates, x in
    column 0
    :param bounds: A tuple of length 4, describing the min x, max x,
    min y and max y coordinates of the square.
    :return: A boolean mask, true is within the square
    """

    # Extact x y coordinates from cloud
    las_xy = cloud.las.points[:,0:2]

    # Create masks for each axis
    x_in = np.logical_and(las_xy[:,0] >= bounds[0] -300,
                          las_xy[:,0] <= bounds[1] + 300)
    y_in = np.logical_and(las_xy[:,1] >= bounds[2] - 300,
                          las_xy[:,1] <= bounds[3] + 300)

    stack = np.stack((x_in, y_in), axis=1)
    in_clip = np.where(np.all(stack, axis=1))

    return(in_clip)

def ray_trace(x, y, poly):
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

def poly_clip(cloud, geometry):
    # Clip to the geometry bounding box
    bbox = geometry.GetEnvelope()
    pre_clip_mask = square_clip(cloud, bbox)
    pre_clip = cloud.las.points[pre_clip_mask]

    # Clip the rest
    geo_json = geometry.ExportToJson()
    geo_json = json.loads(geo_json)
    geo_arr = np.array(geo_json['coordinates'])[0]
    clipped = ray_trace(pre_clip[:,0], pre_clip[:,1], geo_arr)
    return(pre_clip[clipped])



