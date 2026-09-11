---
title: pyfor.cloud
slug: api/pyfor.cloud
description: API reference for the pyfor.cloud module.
sidebar:
  order: 2
---

## Functions

<a id="pyfor.cloud.points_from_columns"></a>

### points_from_columns

```python
points_from_columns(columns)
```

Builds a structured numpy array of points from a mapping of column name to 1D array. This is the only place points are constructed, so every `CloudData.points` array has the same memory layout: one field per dimension, addressable by name.

| Parameter | Type | Description |
| --- | --- | --- |
| `columns` |  | A dictionary of column name to 1D numpy array. |

**Returns:** A structured numpy array.

<a id="pyfor.cloud.points_from_laspy"></a>

### points_from_laspy

```python
points_from_laspy(las, mask = None)
```

Extracts the pyfor point dimensions from a laspy object into a structured numpy array.

| Parameter | Type | Description |
| --- | --- | --- |
| `las` |  | A `laspy.LasData` or `laspy.ScaleAwarePointRecord` object. |
| `mask` |  | An optional boolean mask or array of indices selecting a subset of the points. |

**Returns:** A structured numpy array with one field per dimension present in the file.

<a id="pyfor.cloud.read_polygon"></a>

### read_polygon

```python
read_polygon(path, polygon, chunk_size = 1000000)
```

Reads the points of a `.las` or `.laz` file that fall within a polygon without loading the whole file into memory. Points are read in chunks, pre-filtered by the polygon bounding box, and then tested against the polygon itself.

| Parameter | Type | Description |
| --- | --- | --- |
| `path` |  | The path of the `.las` or `.laz` file to read. |
| `polygon` |  | A `shapely.geometry.Polygon` in the same CRS as the point cloud. |
| `chunk_size` |  | The number of points to read per chunk. |

**Returns:** A structured numpy array of the points within the polygon.

## Classes

<a id="pyfor.cloud.CloudData"></a>

## CloudData

```python
CloudData(points, header)
```

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `header` |  |  |
| `points` |  |  |

<a id="pyfor.cloud.PLYData"></a>

## PLYData

```python
PLYData(points, header)
```

### Methods

<a id="pyfor.cloud.PLYData.write"></a>

#### write

```python
write(path)
```

Writes the object to file. This is a wrapper for `plyfile.PlyData.write`

| Parameter | Type | Description |
| --- | --- | --- |
| `path` |  | The path of the ouput file. |

<a id="pyfor.cloud.LASData"></a>

## LASData

```python
LASData(points, header)
```

### Methods

<a id="pyfor.cloud.LASData.write"></a>

#### write

```python
write(path)
```

Writes the object to file. This is a wrapper for `laspy.LasData.write`, the header stored on the object is copied and its point count and bounds are updated to reflect the points held in memory.

| Parameter | Type | Description |
| --- | --- | --- |
| `path` |  | The path of the ouput file. |

<a id="pyfor.cloud.Cloud"></a>

## Cloud

```python
Cloud(path)
```

The cloud object is an API for interacting with `.las`, `.laz`, and `.ply` files in memory, and is generally the starting point for any analysis with `pyfor`. For a more qualitative assessment of getting started with `Cloud` please see the [user manual](https://github.com/brycefrank/pyfor_manual/blob/master/notebooks/2-ImportsExports.ipynb).

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `filepath` |  |  |
| `name` |  |  |
| `extension` |  |  |
| `data` |  |  |
| `normalized` |  |  |
| `crs` |  |  |

### Properties

<a id="pyfor.cloud.Cloud.convex_hull"></a>

#### convex_hull

Calculates the convex hull of the cloud projected onto a 2d plane, a wrapper for `scipy.spatial.ConvexHull`.

**Returns:** A `shapely.geometry.Polygon` of the convex hull.

### Methods

<a id="pyfor.cloud.Cloud.from_pdal"></a>

#### from_pdal

```python
classmethod from_pdal(ins)
```

Converts a PDAL `ins` argument from a PDAL `filters.python` into a `Cloud` object.

| Parameter | Type | Description |
| --- | --- | --- |
| `ins` |  | The `ins` argument from PDAL. |

<a id="pyfor.cloud.Cloud.grid"></a>

#### grid

```python
grid(cell_size, spec = None)
```

Generates a `Grid` object for the parent object given a cell size. See the documentation for `Grid` for more information.

| Parameter | Type | Description |
| --- | --- | --- |
| `cell_size` |  | The resolution of the plot in the same units as the input file. |
| `spec` |  | An optional `GridSpec`. By default the grid covers the cloud and its origin is snapped to a multiple of the cell size (the target aligned pixels convention), which is what makes rasters from different tiles line up with each other. |

**Returns:** A `Grid` object.

<a id="pyfor.cloud.Cloud.plot"></a>

#### plot

```python
plot(cell_size = 1, cmap = 'viridis', return_plot = False, block = False)
```

Plots a basic canopy height model of the Cloud object. This is mainly a convenience function for `Raster.plot`. More robust methods exist for dealing with canopy height models. Please see the [user manual](https://github.com/brycefrank/pyfor_manual/blob/master/notebooks/3-CanopyHeightModel.ipynb).

| Parameter | Type | Description |
| --- | --- | --- |
| `clip_size` |  | The resolution of the plot in the same units as the input file. |
| `return_plot` |  | If true, returns a matplotlib plt object. |

**Returns:** If return_plot == True, returns matplotlib plt object. Not yet implemented.

<a id="pyfor.cloud.Cloud.plot3d"></a>

#### plot3d

```python
plot3d(dim = 'z', point_size = 1, cmap = 'Spectral_r', max_points = 500000.0, n_bin = 8, plot_trees = False)
```

Plots the three dimensional point cloud using a `Qt` backend. By default, if the point cloud exceeds 5e5 points, then it is downsampled using a uniform random distribution. This is for performance purposes.

| Parameter | Type | Description |
| --- | --- | --- |
| `point_size` |  | The size of the rendered points. |
| `dim` |  | The dimension upon which to color (i.e. "z", "intensity", etc.) |
| `cmap` |  | The matplotlib color map used to color the height distribution. |
| `max_points` |  | The maximum number of points to render. |

<a id="pyfor.cloud.Cloud.normalize"></a>

#### normalize

```python
normalize(cell_size, classified = False, spec = None, kwargs = {})
```

Normalize the cloud using the default Zhang et al. (2003) progressive morphological ground filter. Please see the documentation in `ground_filter.Zhang2003` for more information and keyword argument definitions. If you want to use a pre-computed DEM to normalize, please see `subtract`.

| Parameter | Type | Description |
| --- | --- | --- |
| `cell_size` |  | The resolution of the intermediate bare earth model. |
| `classified` |  | If True and file type is `.las` or `.laz`, uses the points classified as ground (i.e. 2) to construct the intermediate bare earth model. |
| `spec` |  | An optional `GridSpec` for the bare earth model. Passing the same spec for every tile of a project keeps the normalization of those tiles consistent with each other. |

<a id="pyfor.cloud.Cloud.subtract"></a>

#### subtract

```python
subtract(path)
```

Normalize using a pre-computed raster file, i.e. "subtract" the heights from the input raster **in place**. This assumes the raster and the point cloud are in the same coordinate system.

| Parameter | Type | Description |
| --- | --- | --- |
| `path` |  | The path to the raster file, must be in a format supported by `rasterio`. |

**Returns:** 

<a id="pyfor.cloud.Cloud.clip"></a>

#### clip

```python
clip(polygon)
```

Clips the point cloud to the provided shapely polygon using a ray casting algorithm. This method calls `clip.poly_clip` directly. This returns a new `Cloud`.

| Parameter | Type | Description |
| --- | --- | --- |
| `polygon` |  | A `shapely.geometry.Polygon` in the same CRS as the Cloud. |

**Returns:** A new :class:.`Cloud` object clipped to the provided polygon.

<a id="pyfor.cloud.Cloud.filter"></a>

#### filter

```python
filter(min, max, dim)
```

Filters a cloud object for a given dimension **in place**.

| Parameter | Type | Description |
| --- | --- | --- |
| `min` |  | Minimum dimension to retain. |
| `max` |  | Maximum dimension to retain. |
| `dim` |  | The dimension of interest as a string. For example "z". This corresponds to a column label in `self.data.points`. |

<a id="pyfor.cloud.Cloud.chm"></a>

#### chm

```python
chm(cell_size, interp_method = None, pit_filter = None, kernel_size = 3, spec = None)
```

Returns a `Raster` object of the maximum z value in each cell, with optional interpolation (i.e. nan-filling) and pit filter parameters. Currently, only a median pit filter is implemented.

| Parameter | Type | Description |
| --- | --- | --- |
| `cell_size` |  | The cell size for the returned raster in the same units as the parent Cloud or las file. |
| `interp_method` |  | The interpolation method as a string to fill in NA values of the produced canopy height model, one of either "nearest", "cubic", or "linear". This is an argument to `scipy.interpolate.griddata`. |
| `pit_filter` |  | If "median" passes a median filter over the produced canopy height model. |
| `kernel_size` |  | The kernel size of the median filter, must be an odd integer. |
| `spec` |  | An optional `GridSpec` for the canopy height model, see `grid`. |

**Returns:** A `Raster` object of the canopy height model.

<a id="pyfor.cloud.Cloud.standard_metrics"></a>

#### standard_metrics

```python
standard_metrics(heightbreak = 0)
```

<a id="pyfor.cloud.Cloud.write"></a>

#### write

```python
write(path)
```

Write to file. The precise mechanisms of this writing will depend on the file input type. For `.las` files this will be handled by `LASData.write`, for `.ply` files this will be handled by `PLYData.write`.

| Parameter | Type | Description |
| --- | --- | --- |
| `path` |  | The path of the output file. |
