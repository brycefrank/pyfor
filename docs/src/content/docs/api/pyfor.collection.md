---
title: pyfor.collection
slug: api/pyfor.collection
description: API reference for the pyfor.collection module.
sidebar:
  order: 3
---

## Functions

<a id="pyfor.collection.from_dir"></a>

### from_dir

```python
from_dir(las_dir, kwargs = {})
```

Constructs a CloudDataFrame from a directory of las files.

| Parameter | Type | Description |
| --- | --- | --- |
| `las_dir` |  | The directory of las or .laz files. |
| `glob_str` |  | A glob string to select files from the directory. For example: "*.laz" to select .laz files only. |

**Returns:** A CloudDataFrame constructed from the directory of las files.

## Classes

<a id="pyfor.collection.CloudDataFrame"></a>

## CloudDataFrame

```python
CloudDataFrame(args = (), kwargs = {})
```

Implements a data frame structure for processing and managing multiple `Cloud` objects. It is recommended to initialize using the `from_dir` function.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `n_threads` |  |  |
| `tiles` |  |  |

### Properties

<a id="pyfor.collection.CloudDataFrame.bounding_box"></a>

#### bounding_box



**Returns:** A tuple (minx, miny, maxx, maxy) of the bounding box for the entire collection.

### Methods

<a id="pyfor.collection.CloudDataFrame.par_apply"></a>

#### par_apply

```python
par_apply(func, by_file = False, args = None)
```

Apply a function to the collection in parallel. There are two major use cases:

1. **Buffered Tiles**: In the case of buffered tiles, the `func` argument should contain a function that takes two arguments, the first being an aggregated `cloud.Cloud` object, and the second being a `shapely.geometry.Polygon` that describes the bounding box of the aggregated tile. For this case, set `by_file` to False (this is the default).

2. **Raw Files**: In the case of processing raw tiles in parallel, the `func` argument should contain a function that takes only one argument, the absolute file path to the tile at that iteration. For this case, set `by_file` to True.

| Parameter | Type | Description |
| --- | --- | --- |
| `func` |  | A function used to process each tile or raw file (see above). |
| `by_file` |  | Forces `par_apply` to operate on raw files only if True. |
| `args` |  | An optional dictionary of keyword arguments passed to the applying function. |

<a id="pyfor.collection.CloudDataFrame.retile_raster"></a>

#### retile_raster

```python
retile_raster(cell_size, target_tile_size, buffer = 0)
```

A retiling operation that creates raster-compatible sized tiles. Important for creating project-level rasters. Changes `self.tiles` **in place**. Note that the target tile size is approximate, and is rounded to the nearest size that is compatible with the defined cell size.

| Parameter | Type | Description |
| --- | --- | --- |
| `cell_size` |  | The target cell size of the output raster. |
| `target_tile_size` |  | The target tile size of the retiling operation. |
| `buffer` |  | The amount to buffer each input tile. |

<a id="pyfor.collection.CloudDataFrame.reset_tiles"></a>

#### reset_tiles

```python
reset_tiles()
```

Reset the tiles to describe the bounding boxes of each `.las` file **in place**.

<a id="pyfor.collection.CloudDataFrame.grid_spec"></a>

#### grid_spec

```python
grid_spec(cell_size)
```

A `GridSpec` that covers the whole collection, snapped to the cell size.

Passing this spec to the processing of every tile is how per tile rasters are made to line up with each other, because every tile then grids its points on the same lattice rather than on its own extent. It is also what a raster trimmed to a tile boundary needs in order for `Raster.force_extent` to accept that boundary.

| Parameter | Type | Description |
| --- | --- | --- |
| `cell_size` |  | The cell size of the rasters being produced. |

**Returns:** A `GridSpec` covering the collection.

<a id="pyfor.collection.CloudDataFrame.plot"></a>

#### plot

```python
plot(kwargs = {})
```

Plots the bounding boxes of the Cloud objects.

| Parameter | Type | Description |
| --- | --- | --- |
| `**kwargs` |  | Keyword arguments to `geopandas.GeoDataFrame.plot`. |

<a id="pyfor.collection.CloudDataFrame.plot_metrics"></a>

#### plot_metrics

```python
plot_metrics(heightbreak, index = None)
```

Retrieves a set of 29 standard metrics, including height percentiles and other summaries. Intended for use on plot-level point clouds.

| Parameter | Type | Description |
| --- | --- | --- |
| `index` |  | An iterable of indices to set as the output dataframe index. |

**Returns:** A pandas dataframe of standard metrics.

<a id="pyfor.collection.Retiler"></a>

## Retiler

```python
Retiler(cdf)
```

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `cdf` |  |  |

### Methods

<a id="pyfor.collection.Retiler.retile_raster"></a>

#### retile_raster

```python
retile_raster(target_cell_size, original_tile_size, buffer = 0)
```

Creates a retiling grid for a specified target cell size. This creates a list of polygons such that if a raster is constructed from a polygon it will exactly fit inside given the specified target cell size. Useful for creating project level rasters.

| Parameter | Type | Description |
| --- | --- | --- |
| `target_cell_size` |  | The desired output cell size |
| `original_tile_size` |  | The original tile size of the project |
| `buffer` |  | The distance to buffer each new tile to prevent edge effects. |

**Returns:** A list of shapely polygons that correspond to the new grid.

<a id="pyfor.collection.Retiler.retile_buffer"></a>

#### retile_buffer

```python
retile_buffer(buffer)
```

A simple buffering operation.

**Returns:** A list of buffered shapely polygons.
