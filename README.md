# Introduction

**pyfor** is a Python 3 module intended as a tool to assist in the
processing of LiDAR data in the context of forest resources. With pyfor
it is possible to take raw LiDAR data and convert it into a normalized
form for further analysis.

Pyfor was built on top of five major packages:
*  laspy for Python 3.5: reads and writes .las files.
* OGR: handles and creates geospatial information like
 polygons and points.
* GDAL: handles and creates raster data.
* Numpy: fast computational package.
* Pandas: dataframes and manipulation of dataframes.

# Structure

pyfor is organized into three major sections.

1. PointCloud
2. Sampler
3. Metrics