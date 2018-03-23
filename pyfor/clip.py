# Retrieve some data to clip.

import laspy
import gdal
import ogr
import pointcloud2
from numba import jit
from shapely.geometry import Polygon
import numpy as np
import json
import timeit

las1 = laspy.file.File("/home/bryce/Desktop/debug/PC107701LeafOn2010.LAS")
las1 = laspy.file.File("/home/bryce/Desktop/debug/10.las")
las_xy = np.stack([las1.x, las1.y], axis = 0)
#a = pointcloud2.CloudInfo("/home/bryce/Desktop/debug/PC107701LeafOn2010.LAS")

# Load in geometry
plot1 = "/home/bryce/Desktop/debug/plot_107701.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(plot1, 0)
layer = dataSource.GetLayer()
a = layer.GetFeature(0)
firstgeom = a.GetGeometryRef()
b = firstgeom.ExportToJson()

geojson = json.loads(b)
npa = np.array(geojson['coordinates'])[0]
poly = npa

@jit(nopython=True)
def ray_tracing(xy):
    x = xy[0]
    y = xy[1]
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

@jit(nopython=True)
def apply_clip():
    return(np.array([ray_tracing(las_xy[:,i]) for i in range(las_xy.shape[1])]))


t = timeit.timeit('apply_clip()', "from __main__ import apply_clip", number = 5)
print(t)
