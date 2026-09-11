---
title: pyfor.gisexport
slug: api/pyfor.gisexport
description: API reference for the pyfor.gisexport module.
sidebar:
  order: 4
---

## Functions

<a id="pyfor.gisexport.project_indices"></a>

### project_indices

```python
project_indices(indices, raster)
```

Converts indices of an array (for example, those indices that describe the location of a local maxima) to the same space as the input cloud object.

| Parameter | Type | Description |
| --- | --- | --- |
| `indices` |  | The indices to project, an Nx2 matrix of indices where the first column are the rows (Y) and the second column is the columns (X) |
| `raster` |  | An object of type pyfor.rasterizer.Raster |

**Returns:** 

<a id="pyfor.gisexport.array_to_raster"></a>

### array_to_raster

```python
array_to_raster(array, affine, crs, path)
```

Writes a GeoTIFF raster from a numpy array.

| Parameter | Type | Description |
| --- | --- | --- |
| `array` |  | 2D numpy array of cell values |
| `affine` |  | The affine transformation. |
| `crs` |  | A rasterio-compatible coordinate reference (e.g. a proj4 string, WKT, or EPSG code) |
| `path` |  | The output bath of the GeoTIFF |
