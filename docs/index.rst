.. pyfor documentation master file, created by
   sphinx-quickstart on Sat Apr 14 07:55:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyfor: point cloud analysis for forest inventory
================================================

`pyfor <https://github.com/brycefrank/pyfor>`_ is a Python package for processing and manipulating point cloud data for analysis in large scale forest inventory
systems. pyfor is developed with the philsophy of flexibility, and offers solutions for advanced and novice Python users.
This web page contains a user manual and source code documentation.

pyfor is capable of processing large acquisitions of point data in just a few lines of code. Here is an example for
performing a routine normalization for an entire collection of `.las` tiles.

.. code:: python

   import pyfor
   collection = pyfor.collection.from_dir('./my_tiles')

   def normalize(las_path):
      tile = pyfor.cloud.Cloud(las_path)
      tile.normalize()
      tile.write('{}_normalized.las'.format(tile.name))

   collection.par_apply(normalize, by_file=True)

The above example only scratches the surface. Please see the Installation and Getting Started pages to learn more.

.. toctree::
   :maxdepth: 2

   introduction
   installation
   gettingstarted
   topics/index
   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
