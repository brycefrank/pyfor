from pyfor import pointcloud
from pyfor import sampler

cloud1 = pointcloud.CloudInfo(r"C:\pyformaster\samplelas\WA_Olympic_Peninsula_2013_000191\WA_Olympic_Peninsula_2013_000191.las")

cloud1.wkt = """PROJCS["NAD83 / UTM zone 10N",
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



# Index the normalized cloud.
normcloud = pointcloud.CloudInfo(r"C:\pyformaster\samplelas\WA_Olympic_Peninsula_2013_000191\WA_Olympic_Peninsula_2013_000191.las")
normcloud.grid_constructor(10)
normcloud.cell_sort()


normcloud.dem_path = r"C:\pyformaster\pyfordata\Ancillary_Data\mydata\mydem.tiff"
normcloud.normalize()


sample1 = sampler.Sampler(normcloud)
sample1.plot_shp = r"C:\pyformaster\pyfordata\Ancillary_Data\gis\newplots1.shp"

sample1.grid_path = r"C:\pyformaster\pyfordata\Ancillary_Data\gis\mygrid.shp"

sample1.clip_plots(r"C:\pyformaster\pyfordata\Ancillary_Data\mydata\plots_clip")
#
# print(sample1.extract_points())