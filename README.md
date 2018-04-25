<p align="center">
  <img src="https://github.com/brycefrank/pyfor/blob/pdal-u/docs/tile.png" width="400">
</p>

[![Documentation Status](https://readthedocs.org/projects/pyfor-pdal-u/badge/?version=pdal-u)](http://pyfor-pdal-u.readthedocs.io/en/pdal-u/?badge=pdal-u)[![Build Status](https://travis-ci.org/brycefrank/pyfor.svg?branch=pdal-u)](https://travis-ci.org/brycefrank/pyfor)[![Coverage Status](https://coveralls.io/repos/github/brycefrank/pyfor/badge.svg?branch=pdal-u)](https://coveralls.io/github/brycefrank/pyfor?branch=pdal-u)

**pyfor** is a Python module intended as a tool to assist in the processing of point cloud data in the context of forest inventory. It offers functions that convert raw point cloud data to usable information about forested landscapes using an object oriented (OOP) framework accessible for advanced and novice users of Python. pyfor aims to provide a cross platform means to interactively process point cloud data, as well as efficient ways to batch process large acquisitions.

pyfor is currently undergoing major revisions from its infant stage to incorporate several packages that speed up processing time and allow for visualizations of point cloud data. The expected release for a stable working version (0.2.0) is **May 2018**, but this branch will provide rolling release updates for anyone interested in trying the package early.

## Samples

These samples are a work in progress, but demonstrate some of the package capabilities.

- [File Input and Plotting](https://github.com/brycefrank/pyfor/blob/pdal-u/samples/ImportsExports.ipynb)
- [Normalization](https://github.com/brycefrank/pyfor/blob/pdal-u/samples/Normalization.ipynb)
- [Canopy Height Models](https://github.com/brycefrank/pyfor/blob/pdal-u/samples/CanopyHeightModel.ipynb)

## Installation

For installation I highly recommend looking into setting up [miniconda](https://conda.io/miniconda.html) for your system before beginning. pyfor depends on many packages that are otherwise tricky and difficult to install (especially gdal and its bindings), and conda provides a quick and easy way to manage many different Python environments on your system simultaneously.

Note that the following installation procedures will install the rolling release version of pyfor (i.e. this branches' source files). I develop actively on this branch and push commits daily, so beware of intermittent bugs when updating. More stable releases are forthcoming. 

The following bash commands will install this branch of pyfor. It requires installation of miniconda (see above). This will install all of the prerequisites in that environment, named `pyfor_env`. pyfor depends on a lot of heavy libraries, so expect construction of the environment to take a little time.

```bash
git clone https://github.com/brycefrank/pyfor.git
cd pyfor
git checkout pdal-u
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

[under construction]

An early version of the documentation is located [here](http://pyfor-pdal-u.readthedocs.io/en/pdal-u/). Beware of many formatting issues yet to be fixed.

An early collection of samples is located [here](https://github.com/brycefrank/pyfor/tree/pdal-u/samples)

## Features

Below is a list of features planned for the 0.2 stable release. 0.2 is intended to be adequate for processing and visualizing individual point cloud tiles.

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

And forthcoming in the 0.3 stable release, this release is intended for batch processing of many tiles by providing multiprocessor and cluster support.

- [ ] Batch processing
	- [ ] Multiprocessor support
	- [ ] Cluster support
- [ ] Voxelization methods
- [ ] More tree detection methods

## Goals

- Maintain a purely Python code base
- Maintain compatibility between operating systems via conda
- Support processing for distributed systems and multicore processors
- Implement new

## Collaboration & Requests

If you would like to contribute, especially those experienced with `numba`, `numpy`, `gdal`, `ogr` and `pandas`, please contact me at bfrank70@gmail.com For a list of to do items before our first release, please see the [Working Prototype](https://github.com/brycefrank/pyfor/projects/3) page.

I am also willing to implement features on request. Feel free to [open an issue](https://github.com/brycefrank/pyfor/issues) with your request or email me at the adress above.
