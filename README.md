# Introduction

**PyFor** is a Python 3 module intended as a tool to assist in the
processing of LiDAR data in the context of forest resources. With PyFor
it is possible to take raw LiDAR data and convert it into a normalized
form for further analysis. The main repository is located at
https://github.com/brycefrank/PyFor

This package is developed and maintained by Bryce Frank.

# Structure

PyFor was built on top of five major packages:
*  laspy for Python 3.5: reads and writes .las files.
* OGR: handles and creates geospatial information like
 polygons and points.
* GDAL: handles and creates raster data.
* Numpy: fast computational package.
* Pandas: dataframes and manipulation of dataframes.

# Change Log

v 0.1, 2-28-2017 -- Initial release.

# TODO

In no particular order:

* Documentation
* Register package on pip
* Develop additional ground filter functions and bindings
* Implement tests
* Canopy Height Model
* Grid Metrics
* PEP8 - Method Variable Refactoring