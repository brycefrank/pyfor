.. pyfor documentation master file, created by
   sphinx-quickstart on Sat Apr 14 07:55:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyfor
=================================

This is the documentation repository for **pyfor**, a Python package for point cloud data processing in the context of forest inventory. Our GitHub page is located `here <https://github.com/brycefrank/pyfor>`_. This website provides documentation of individual classes and functions as well as more general documents about the structure of pyfor.

Classes
=======

pyfor is written using an OOP framework. The classes listed below make up the most of the package high-level functionality.

* `Cloud <source/pyfor.html#pyfor.cloud.Cloud>`_ - represents the point cloud itself.
* `CloudData <source/pyfor.html#pyfor.cloud.CloudData>`_ - handles the point cloud data manipulations (mostly an internal class).
* `Grid <source/pyfor.html#pyfor.rasterizer.Grid>`_ - represents the point cloud as separated into grid cells (many points per grid cell).
* `Raster <source/pyfor.html#pyfor.rasterizer.Raster>`_ - represents the point cloud as a two-dimensional raster (one value per grid cell).
* `CloudDataFrame <source/pyfor.html#pyfor.collection.CloudDataFrame>`_ - represents a collection of Cloud objects.

Functions
=========

The beating heart of pyfor is its collections of functions, these handle the lower level processing tasks. They are located in a few different scripts:

* `clip_funcs.py <source/pyfor.html#module-pyfor.clip_funcs.html>`_ - holds functions for clipping point cloud data.
* `gisexport.py <source/pyfor.html#module-pyfor.gisexport.html>`_ - holds functions for writing to GIS data types, mostly implemented via rasterio, fiona and geopandas.
* `filter.py <source/pyfor.html#module-pyfor.filter.html>`_ - holds ground filtering and related functions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
