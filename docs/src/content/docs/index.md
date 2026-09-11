---
title: pyfor
description: Python tools for processing point cloud data in large scale forest inventory systems.
template: splash
hero:
  tagline: A Python package for processing and manipulating point cloud data for analysis in large scale forest inventory systems.
  image:
    file: ../../assets/logo.png
    alt: pyfor
  actions:
    - text: Getting Started
      link: /pyfor/gettingstarted/
      icon: right-arrow
      variant: primary
    - text: GitHub
      link: https://github.com/brycefrank/pyfor
      icon: external
      variant: minimal
---

**pyfor** is a Python package for processing and manipulating point cloud data for analysis in
large scale forest inventory systems. It is developed with the philosophy of flexibility, and
offers solutions for advanced and novice Python users. This site contains a user manual and the
[API reference](/pyfor/api/).

pyfor is capable of processing large acquisitions of point data in just a few lines of code. Here
is an example of a routine normalization for an entire collection of `.las` tiles:

```python
import pyfor
collection = pyfor.collection.from_dir("./my_tiles")

def normalize(las_path):
    tile = pyfor.cloud.Cloud(las_path)
    tile.normalize()
    tile.write("{}_normalized.las".format(tile.name))

collection.par_apply(normalize, by_file=True)
```

The above example only scratches the surface. See [Installation](/pyfor/installation/) and
[Getting Started](/pyfor/gettingstarted/) to learn more.
