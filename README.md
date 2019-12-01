<p align="center">
  <img src="docs/logo.png" width="100"><br>
  <b>pyfor</b><br><br>
  <a href="http://brycefrank.com/pyfor">Documentation</a> |
  <a href="https://github.com/brycefrank/pyfor/blob/master/CHANGELOG.md">Changelog</a> |
  <a href="https://github.com/brycefrank/pyfor/issues/new">Request a Feature</a> |
  <a href="https://github.com/brycefrank/pyfor/projects/12">Road Map</a>
  <br>
  <img src="https://camo.githubusercontent.com/033f1149793306148313011a8777f72724800836/68747470733a2f2f7472617669732d63692e6f72672f62727963656672616e6b2f7079666f722e7376673f6272616e63683d6d6173746572">
  <img src="https://coveralls.io/repos/github/brycefrank/pyfor/badge.svg?branch=master">
</p>

**pyfor** is a Python package that assists in the processing of point cloud data in the context of forest inventory. 
This includes manipulation of point data, support for analysis, and a
memory optimized API for managing large collections of tiles.

## Release Status

Current Release: 0.3.6

Release Date: December 1st, 2019.

Release Status: 0.3.6 is an adolescent LiDAR data processing package adequate for single tile processing and large acqusitions.

## What Does pyfor Do?

- [Normalization](http://brycefrank.com/pyfor/html/topics/normalization.html)
- [Canopy Height Models](http://brycefrank.com/pyfor/html/topics/canopyheightmodel.html)
- [Ground Filtering](http://brycefrank.com/pyfor/html/api/pyfor.ground_filter.html)
- [Clipping](http://brycefrank.com/pyfor/html/topics/clipping.html)
- [Large Acquisition Processing](http://brycefrank.com/pyfor/html/advanced/handlinglargeacquisitions.html)

and many other tasks. See the [documentation](http://brycefrank.com/pyfor) for examples and applications.

What about tree segmentation? Please see pyfor's sister package [`treeseg`](https://github.com/brycefrank/treeseg) which
is a standalone package for tree segmentation and detection.

## Installation

[miniconda](https://conda.io/miniconda.html) or Anaconda is required for your system before beginning. pyfor depends on many packages that are otherwise tricky and difficult to install (especially gdal and its bindings), and conda provides a quick and easy way to manage many different Python environments on your system simultaneously.

As of October 14th, 2019, we are proud to announce that `pyfor` is available on `conda-forge`, greatly simplifying the installation process:

```
conda install -c conda-forge pyfor 
```

## Collaboration & Requests

If you would like to contribute, especially those experienced with `numba`, `numpy`, `gdal`, `ogr` and `pandas`, please contact me at bfrank70@gmail.com 

I am also willing to implement features on request. Feel free to [open an issue](https://github.com/brycefrank/pyfor/issues) with your request or email me at the address above.

pyfor will always remain a free service. Its development takes time, energy and a bit of money to maintain source code and host documentation. If you are so inclined, donations are accepted at the donation button at the top of the readme.

