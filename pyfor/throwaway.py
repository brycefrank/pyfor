import laspy
import numpy as np
import pointcloud2
import ogr
import clip_funcs

# Load point cloud
a = pointcloud2.Cloud("/home/bryce/Desktop/pyfor_test_data/PC_001.las")


# Load geometry
poly1 = "/home/bryce/Desktop/pyfor_test_data/pc_001_clip_a.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(poly1, 0)
layer = dataSource.GetLayer()
feat = layer.GetFeature(0)
geom = feat.GetGeometryRef()

#bbox = geom.GetEnvelope()
#print(type(bbox))
#b = a.clip(bbox)
#b.plot()
print(type(a.las.header))
#a.clip(geom).plot(0.2)

#out = laspy.file.File("test.las", header = a.las.header, mode = "w")
#out.points = a.las.points[mask]
#out_c = pointcloud2.Cloud(out)
#out_c.plot(1)


