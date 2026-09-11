---
title: pyfor.rasterizer
slug: api/pyfor.rasterizer
description: API reference for the pyfor.rasterizer module.
sidebar:
  order: 7
---

## Functions

<a id="pyfor.rasterizer.sample_array"></a>

### sample_array

```python
sample_array(array, bins_x, bins_y)
```

Looks up the value of a raster array for each point from its column and row bins.

| Parameter | Type | Description |
| --- | --- | --- |
| `array` |  | A 2D numpy array indexed as ``array[bins_y, bins_x]``. |
| `bins_x` |  | A 1D integer array of the column bin of each point. |
| `bins_y` |  | A 1D integer array of the row bin of each point. |

**Returns:** A 1D float array of values, NaN where the bin of a point is outside of `array`.

<a id="pyfor.rasterizer.reduce_cells"></a>

### reduce_cells

```python
reduce_cells(values, cells, n_cells, func)
```

Reduces the values of every occupied cell of a grid with `func`.

| Parameter | Type | Description |
| --- | --- | --- |
| `values` |  | A 1D array of values, one per point. |
| `cells` |  | A 1D integer array of the cell id of each point. |
| `n_cells` |  | The total number of cells, occupied or not. |
| `func` |  | Either the name of a reduction ("count", "sum", "mean", "min", "max", "std", "var"), a numpy function (i.e. `np.max`), or any callable that reduces a 1D array of values to a scalar (i.e. `np.percentile` or a lambda). |

**Returns:** A tuple (cell_ids, results) of the occupied cells, ordered by ascending cell id.

<a id="pyfor.rasterizer.percentile_cells"></a>

### percentile_cells

```python
percentile_cells(values, cells, percentiles)
```

Computes percentiles of the values of every occupied cell. This is a vectorized equivalent of calling `numpy.percentile`, with its default linear interpolation, on the values of each cell, which is the most expensive reduction pyfor performs.

| Parameter | Type | Description |
| --- | --- | --- |
| `values` |  | A 1D array of values, one per point. |
| `cells` |  | A 1D integer array of the cell id of each point. |
| `percentiles` |  | A sequence of percentiles between 0 and 100. |

**Returns:** A tuple (cell_ids, results) where results has shape ``(len(percentiles), len(cell_ids))``.

<a id="pyfor.rasterizer.cell_bins"></a>

### cell_bins

```python
cell_bins(points, spec)
```

The column and row bin of every point on a grid.

Cells are half open and counted from the bottom left corner, which is how GDAL and rasterio locate a point in a raster: a point at ``origin_y + k * cell_size`` is in row ``k``, so it is in the cell above the line when the raster is read north up. Bins are clipped into the grid, which only affects points lying exactly on the far edge of the extent; GDAL would consider those outside the raster and PDAL drops them.

| Parameter | Type | Description |
| --- | --- | --- |
| `points` |  | A structured numpy array of points, as held in `CloudData.points`. |
| `spec` |  | A `GridSpec`. |

**Returns:** A tuple (bins_x, bins_y) of 1D integer arrays, counted from the top left of the raster.

## Classes

<a id="pyfor.rasterizer.GridSpec"></a>

## GridSpec

```python
GridSpec(origin_x, origin_y, cell_size, n, m)
```

The grid a raster is computed on: the bottom left corner, the cell size, and the size in cells.

| Parameter | Type | Description |
| --- | --- | --- |
| `origin_x` |  | The x coordinate of the left edge of the grid. |
| `origin_y` |  | The y coordinate of the bottom edge of the grid. |
| `cell_size` |  | The size of a cell, in the units of the point cloud. |
| `n` |  | The number of cells in the x direction. |
| `m` |  | The number of cells in the y direction. |

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `origin_x` | float |  |
| `origin_y` | float |  |
| `cell_size` | float |  |
| `n` | int |  |
| `m` | int |  |

### Properties

<a id="pyfor.rasterizer.GridSpec.bounds"></a>

#### bounds



**Returns:** A tuple (min_x, min_y, max_x, max_y) of the grid.

<a id="pyfor.rasterizer.GridSpec.shape"></a>

#### shape



**Returns:** The shape (m, n) of a raster on this grid.

<a id="pyfor.rasterizer.GridSpec.affine"></a>

#### affine



**Returns:** The rasterio affine transformation of the grid, north up.

### Methods

<a id="pyfor.rasterizer.GridSpec.covering"></a>

#### covering

```python
classmethod covering(min_x, min_y, max_x, max_y, cell_size)
```

The grid that covers an extent, with its origin snapped to a multiple of the cell size.

Snapping is the target aligned pixels convention (`gdal_translate -tap`, and the default in terra and lidR). It is what makes rasters from neighbouring tiles line up: every snapped origin sits on the same lattice of multiples of ``cell_size``, so two tiles of a project describe the same cells, and a raster can be mosaicked without resampling. Tools whose grid defaults to the extent of the data, pyfor before this and PDAL's ``writers.gdal``, produce rasters that cannot be compared or mosaicked cell by cell.

<a id="pyfor.rasterizer.GridSpec.covers"></a>

#### covers

```python
covers(min_x, min_y, max_x, max_y)
```

**Returns:** True if the grid covers the given extent.

<a id="pyfor.rasterizer.Grid"></a>

## Grid

```python
Grid(cloud, cell_size = None, spec = None)
```

The Grid object is a representation of a point cloud that has been sorted into X and Y dimensional bins. From the Grid object we can derive other useful products, most importantly, `Raster` objects.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `cloud` |  |  |
| `spec` |  |  |
| `cell_size` |  |  |
| `cell_ids` |  |  |

### Properties

<a id="pyfor.rasterizer.Grid.n_cells"></a>

#### n_cells



**Returns:** The total number of cells in the grid, occupied or not.

<a id="pyfor.rasterizer.Grid.empty_cells"></a>

#### empty_cells

Retrieves the cells with no returns in self.data return: An N x 2 numpy array where each row cooresponds to the [y x] coordinate of the empty cell.

### Methods

<a id="pyfor.rasterizer.Grid.reduce"></a>

#### reduce

```python
reduce(func, dim, mask = None)
```

Reduces a dimension of the parent cloud for every occupied cell.

| Parameter | Type | Description |
| --- | --- | --- |
| `func` |  | A function to reduce the values of a cell, see `reduce_cells`. |
| `dim` |  | The dimension (i.e. column name of the points) to reduce. |
| `mask` |  | An optional boolean mask, only the selected points are reduced. |

**Returns:** A tuple (bins_y, bins_x, values) of the occupied cells.

<a id="pyfor.rasterizer.Grid.cell_ranks"></a>

#### cell_ranks

```python
cell_ranks()
```

The ordinal of each point within its own cell, in the order the points are held by the cloud.

**Returns:** A 1D integer array, zero for the first point of every cell.

<a id="pyfor.rasterizer.Grid.cell_counts"></a>

#### cell_counts

```python
cell_counts(mask = None)
```

Counts the points of the parent cloud in every cell.

| Parameter | Type | Description |
| --- | --- | --- |
| `mask` |  | An optional boolean mask selecting the points to count. |

**Returns:** A 1D integer array of length `n_cells`.

<a id="pyfor.rasterizer.Grid.cell_values"></a>

#### cell_values

```python
cell_values(func, dim, mask = None)
```

Reduces a dimension of the parent cloud for every cell of the grid.

| Parameter | Type | Description |
| --- | --- | --- |
| `func` |  | A function to reduce the values of a cell, see `reduce_cells`. |
| `dim` |  | The dimension (i.e. column name of the points) to reduce. |
| `mask` |  | An optional boolean mask, only the selected points are reduced. |

**Returns:** A 1D array of length `n_cells`, NaN where a cell is empty. The value of the cell (bins_y, bins_x) is at index ``bins_y * n + bins_x``.

<a id="pyfor.rasterizer.Grid.cell_percentiles"></a>

#### cell_percentiles

```python
cell_percentiles(dim, percentiles)
```

Computes percentiles of a dimension of the parent cloud for every occupied cell.

| Parameter | Type | Description |
| --- | --- | --- |
| `dim` |  | The dimension (i.e. column name of the points) to reduce. |
| `percentiles` |  | A sequence of percentiles between 0 and 100. |

**Returns:** A tuple (cell_ids, results) where results has shape ``(len(percentiles), len(cell_ids))``.

<a id="pyfor.rasterizer.Grid.percentile_raster"></a>

#### percentile_raster

```python
percentile_raster(dim, percentile)
```

A raster of a single percentile of a dimension of the parent cloud.

| Parameter | Type | Description |
| --- | --- | --- |
| `dim` |  | The dimension (i.e. column name of the points) to reduce. |
| `percentile` |  | A percentile between 0 and 100. |

**Returns:** A `Raster` object.

<a id="pyfor.rasterizer.Grid.expand"></a>

#### expand

```python
expand(values)
```

Casts per cell values back onto the points of the parent cloud.

| Parameter | Type | Description |
| --- | --- | --- |
| `values` |  | A 1D array of length `n_cells`, i.e. the result of `cell_values`. |

**Returns:** A 1D array with the value of the cell of each point.

<a id="pyfor.rasterizer.Grid.raster"></a>

#### raster

```python
raster(func, dim, kwargs = {})
```

Generates an m x n matrix with values as calculated for each cell in func. This is a raw array without missing cells interpolated. See self.interpolate for interpolation methods.

| Parameter | Type | Description |
| --- | --- | --- |
| `func` |  | A function string, i.e. "max" or a function itself, i.e. `np.max`. This function must be able to take a 1D array of the given dimension as an input and produce a single value as an output. This single value will become the value of each cell in the array. |
| `dim` |  | A dimension to calculate on. |

**Returns:** A 2D numpy array where the value of each cell is the result of the passed function.

<a id="pyfor.rasterizer.Grid.interpolate"></a>

#### interpolate

```python
interpolate(func, dim, interp_method = 'nearest', mask = None)
```

Interpolates missing cells in the grid. This function uses scipy.griddata as a backend. Please see documentation for that function for more details.

| Parameter | Type | Description |
| --- | --- | --- |
| `func` |  | The function (or function string) to calculate an array on the gridded data. |
| `dim` |  | The dimension (i.e. column name of self.cells) to cast func onto. |
| `interp_method` |  | The interpolation method call for scipy.griddata, one of any: "nearest", "cubic", "linear" |
| `mask` |  | An optional boolean mask, only the selected points are interpolated. |

**Returns:** An interpolated array.

<a id="pyfor.rasterizer.Grid.metrics"></a>

#### metrics

```python
metrics(func_dict, as_raster = False)
```

Calculates summary statistics for each grid cell in the Grid.

| Parameter | Type | Description |
| --- | --- | --- |
| `func_dict` |  | A dictionary containing keys corresponding to the columns of self.data and values that correspond to the functions to be called on those columns. |

**Returns:** A pandas dataframe with the aggregated metrics.

<a id="pyfor.rasterizer.Grid.standard_metrics"></a>

#### standard_metrics

```python
standard_metrics(heightbreak = 0)
```

<a id="pyfor.rasterizer.ImportedGrid"></a>

## ImportedGrid

```python
ImportedGrid(path, cloud)
```

ImportedGrid is used to normalize a parent cloud object with an arbitrary raster file. The grid is the grid of that raster, so the values of the raster can be looked up by the cell of a point.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `array` |  |  |
| `bounds` |  |  |
| `crs` |  |  |
| `spec` |  |  |
| `cloud` |  |  |
| `cell_size` |  |  |
| `cell_ids` |  |  |

<a id="pyfor.rasterizer.Raster"></a>

## Raster

```python
Raster(array, grid)
```

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `grid` |  |  |
| `cell_size` |  |  |
| `array` |  |  |

### Methods

<a id="pyfor.rasterizer.Raster.sample"></a>

#### sample

```python
sample(bins_x, bins_y)
```

Looks up the value of the raster for each point from its column and row bins.

| Parameter | Type | Description |
| --- | --- | --- |
| `bins_x` |  | A 1D integer array of the column bin of each point. |
| `bins_y` |  | A 1D integer array of the row bin of each point. |

**Returns:** A 1D float array of values, NaN where the bin of a point is outside of the raster.

<a id="pyfor.rasterizer.Raster.force_extent"></a>

#### force_extent

```python
force_extent(bbox)
```

Sets `self._affine` and `self.array` to a forced bounding box. Useful for trimming edges off of rasters when processing buffered tiles. This operation is done in place.

The bounding box has to fall on the cells of the raster, otherwise the array cannot be trimmed to it without moving the data to different cells than the affine describes. A `ValueError` is raised rather than quietly rounding, which would leave an array labelled with a grid it is not on.

| Parameter | Type | Description |
| --- | --- | --- |
| `bbox` |  | Coordinates of output raster as a tuple (min_x, max_x, min_y, max_y) |

<a id="pyfor.rasterizer.Raster.plot"></a>

#### plot

```python
plot(cmap = 'viridis', block = False, return_plot = False)
```

Default plotting method for the Raster object.

<a id="pyfor.rasterizer.Raster.pit_filter"></a>

#### pit_filter

```python
pit_filter(kernel_size)
```

Filters pits in the raster. Intended for use with canopy height models (i.e. grid(0.5).interpolate("max", "z"). This function modifies the raster array **in place**.

| Parameter | Type | Description |
| --- | --- | --- |
| `kernel_size` |  | The size of the kernel window to pass over the array. For example 3 -&gt; 3x3 kernel window. |

<a id="pyfor.rasterizer.Raster.write"></a>

#### write

```python
write(path)
```

Writes the raster to a geotiff. Requires the Cloud.crs attribute to be filled by a projection string (ideally wkt or proj4).

| Parameter | Type | Description |
| --- | --- | --- |
| `path` |  | The path to write to. |
