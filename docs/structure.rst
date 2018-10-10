The structure of pyfor
============================

This document provides an in-depth discussion of the structure of pyfor.

Introduction
============

pyfor is unique from other forest inventory LiDAR processing packages in that it is built on top of an object oriented (OOP) framework. Python is not stricly an OOP programming language, but allows for its implementation in a flexible way. pyfor takes advantage of this attractive feature of Python to offer a natural way of thinking about LiDAR as a collection of objects. Each object flows into the other through the process of LiDAR data analysis. Using pyfor means being comfortable with this framework and knowledgeable about what each of the following objects can do. This document is a primer on grasping this philosophy.

There are four main classes that define pyfor. They are:

1. Cloud
2. CloudDataFrame
3. Grid
4. Raster

Cloud
-----

The Cloud class represents the point cloud data in its raw format, a list of x, y and z coordinates (and some other fields like intensity and the like). The Cloud object can be plotted in a variety of ways, including 2D and 3D plots (see .plot, .plot3d, .iplot3d). The Cloud is generally considered the starting point of LiDAR analysis, from this object we can produce other objects, specifically the Raster and Grid.

Grid
----

The Grid can be considered the next step in producing products from our Cloud. The Grid assigns each point of the Cloud to a grid cell. This process allows us to summarize information about the points in each grid cell.

Raster
------

Once we have decided how to summarize our grid cells, we produce a Raster. The Raster is a geo-referenced numpy array where the value of each cell is some summarization of the points within that cell. The most common implementation of a Raster is the canopy height model (CHM) that is a summary of the highest points within each grid cell, but other types are possible, such as bare earth models (BEM) and grid metrics. We can write Rasters to a GeoTIFF format using Raster.write().

CloudDataFrame
--------------

At some point we will want to interact with our LiDAR tiles in toto. CloudDataFrame is a collection of Cloud objects that allows for efficient analysis and manupulation of many Cloud objects. Because Cloud is considered the most base of all classes in pyfor, CloudDataFrame is our portal to all the products we desire on a mass scale.
