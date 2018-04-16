.. pyfor documentation master file, created by
   sphinx-quickstart on Sat Apr 14 07:55:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyfor
=================================

This is the documentation repository for **pyfor**, a Python package for point cloud data processing in the context of forest inventory.Our GitHub page is located `here <https://github.com/brycefrank/pyfor/tree/pdal-u>`_. Please refer to that page (specifically the Wiki) if you are interested in a "higher-level" user manual, installation instructions and the like.

However, if you are interested in nitty-gritty documentation of functions, this is the place to be. Consider this the last stop before reading source code.


Classes
=======

pyfor is written using an OOP framework. The classes listed below make up the most of the package high-level functionality.

* `Cloud <source/pyfor.cloud.html#pyfor.cloud.Cloud>`_ - represents the point cloud itself.
* `CloudData <source/pyfor.cloud.html#pyfor.cloud.CloudData>`_ - handles the point cloud data manipulations (mostly an internal class).
* `Grid <source/pyfor.rasterizer.html#pyfor.rasterizer.Grid>`_ - represents the point cloud as separated into grid cells (many points per grid cell).
* `Raster <source/pyfor.rasterizer.html#pyfor.rasterizer.Raster>`_ - represents the point cloud as a two-dimensional raster (one value per grid cell).

Functions
=========

The beating heart of pyfor is its collections of functions, these handle the lower level processing tasks. They are located in a few different scripts:

* `clip_funcs.py <source/pyfor.clip_funcs.html>`_ - holds functions for clipping point cloud data.
* `gisexport.py <source/pyfor.gisexport.html>`_ - holds functions for writing to GIS file types, mostly a wrapper for GDAL.
* `filter.py <source/pyfor.filter.html>`_ - holds ground filtering and related functions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
