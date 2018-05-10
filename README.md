<p align="center">
  <img src="docs/tile.png" width="400">
</p>

[![Documentation Status](https://readthedocs.org/projects/pyfor/badge/?version=latest)](http://pyfor.readthedocs.io/en/latest/?badge=latest)[![Build Status](https://travis-ci.org/brycefrank/pyfor.svg?branch=master)](https://travis-ci.org/brycefrank/pyfor)[![Coverage Status](https://coveralls.io/repos/github/brycefrank/pyfor/badge.svg?branch=master)](https://coveralls.io/github/brycefrank/pyfor?branch=master)


**pyfor** is a Python package that assists in the processing of point cloud data in the context of forest inventory. It offers functions that convert raw point cloud data to usable information about forested landscapes using an object oriented (OOP) framework accessible for advanced and novice users of Python. pyfor aims to provide a cross platform means to interactively process point cloud data, as well as efficient ways to batch process large acquisitions.

The current release is 0.2.1 (this branch). 0.2.1 is adequate for processing single tiles and serves as the foundation for future updates. There is a dedicated changelog [here](CHANGELOG.md)

A minor update rolling release branch is located on [0.2.2](http://github.com/brycefrank/pyfor/tree/0.2.2). 0.2.2 will only improve on single tile functionality (tree detection, bug fixes, etc). Please check the changelog there before submitting issues.

An upcoming release, [0.3.0](http://github.com/brycefrank/pyfor/tree/0.3.0),  will focus on processing large acquisitions, and is slated for release on this branch in **August 2018**.

## Samples

These samples are a work in progress, but demonstrate some of the package capabilities.

- [File Input and Plotting](https://github.com/brycefrank/pyfor/blob/master/samples/ImportsExports.ipynb)
- [Normalization](https://github.com/brycefrank/pyfor/blob/master/samples/Normalization.ipynb)
- [Canopy Height Models](https://github.com/brycefrank/pyfor/blob/master/samples/CanopyHeightModel.ipynb)
- [Clipping](https://github.com/brycefrank/pyfor/blob/master/samples/Clipping.ipynb)
- [Watershed Segmentation](https://github.com/brycefrank/pyfor/blob/master/samples/WatersgedSegmentation.ipynb)
- [Grid Metrics](https://github.com/brycefrank/pyfor/blob/master/samples/GridMetrics.ipynb)

## Installation

For installation I highly recommend looking into setting up [miniconda](https://conda.io/miniconda.html) for your system before beginning. pyfor depends on many packages that are otherwise tricky and difficult to install (especially gdal and its bindings), and conda provides a quick and easy way to manage many different Python environments on your system simultaneously.

The following bash commands will install this branch of pyfor. It requires installation of miniconda (see above). This will install all of the prerequisites in that environment, named `pyfor_env`. pyfor depends on a lot of heavy libraries, so expect construction of the environment to take a little time.

```bash
git clone https://github.com/brycefrank/pyfor.git
cd pyfor
conda env create -f environment.yml

# For Linux / macOS:
source activate pyfor_env

# For Windows:
activate pyfor_env

pip install .
```

Following these commands, pyfor should load in the activated Python shell.

```python
import pyfor
```

If you see no errors, you are ready to process!

## Getting Started

An early collection of samples is located [here](https://github.com/brycefrank/pyfor/tree/master/samples). These demonstrate some basic tasks.

An early version of the documentation is located [here](http://pyfor-pdal-u.readthedocs.io/en/pdal-u/). This provides specific documentation for each class and function.

## Features

Below is a list of features, complete and otherwise.

- [X] Ground filter
- [X] Normalization
- [X] Rasterization
	- [X] Raster input and output
- [X] Interactive 2d & 3d plotting via Jupyter
	- [X] Point cloud plotting
	- [X] Raster plotting
- [X] Grid metrics extraction
- [X] Watershed segmentation
- [X] Canopy height model
	- [X] Median pit filter
	- [ ] Pit free algorithm (Chen et al. 2017)
- [X] Clipping point clouds
- [ ] Area-based approach workflows
- [ ] Batch processing
	- [ ] Multiprocessor support
	- [ ] Cluster support
- [ ] Voxelization methods
- [ ] More tree detection methods

## Goals

- Maintain a purely Python code base
- Maintain compatibility between operating systems via conda
- Support processing for distributed systems and multicore processors

## Collaboration & Requests

If you would like to contribute, especially those experienced with `numba`, `numpy`, `gdal`, `ogr` and `pandas`, please contact me at bfrank70@gmail.com 

I am also willing to implement features on request. Feel free to [open an issue](https://github.com/brycefrank/pyfor/issues) with your request or email me at the address above.

