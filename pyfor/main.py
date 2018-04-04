import gdal
import numpy as np
import laspy
import gisexport
import pointcloud2

cloud1 = pointcloud2.Cloud("/home/bryce/Desktop/pyfor_test_data/plot_tiles/PC107701LeafOn2010.LAS")
cloud1_grid = cloud1.grid(1)


write_array = cloud1_grid.array("max", "z")

wkt = """PROJCS["NAD83 / UTM zone 10N",
    GEOGCS["NAD83",
        DATUM["North_American_Datum_1983",
            SPHEROID["GRS 1980",6378137,298.257222101,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6269"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4269"]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-123],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    AUTHORITY["EPSG","26910"],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]"""

gisexport.array_to_raster(write_array, 1, 472137, 5015683, wkt, "/home/bryce/Desktop/testc.tif")

print(write_array.shape)
