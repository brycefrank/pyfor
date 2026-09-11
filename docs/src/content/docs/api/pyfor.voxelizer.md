---
title: pyfor.voxelizer
slug: api/pyfor.voxelizer
description: API reference for the pyfor.voxelizer module.
sidebar:
  order: 8
---

## Classes

<a id="pyfor.voxelizer.VoxelGrid"></a>

## VoxelGrid

```python
VoxelGrid(cloud, cell_size)
```

A 3 dimensional grid representation of a point cloud. This is analagous to the rasterizer.Grid class, but with three axes instead of two. VoxelGrids are generally used to produce VoxelRaster objects.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `cell_size` |  |  |
| `cloud` |  |  |
| `m` |  |  |
| `n` |  |  |
| `p` |  |  |
| `bins_x` |  |  |
| `bins_y` |  |  |
| `bins_z` |  |  |
| `cell_ids` |  |  |

### Properties

<a id="pyfor.voxelizer.VoxelGrid.n_cells"></a>

#### n_cells



**Returns:** The total number of voxels, occupied or not.

### Methods

<a id="pyfor.voxelizer.VoxelGrid.voxel_raster"></a>

#### voxel_raster

```python
voxel_raster(func, dim)
```

Creates a 3 dimensional voxel raster, analagous to rasterizer.Grid.raster.

| Parameter | Type | Description |
| --- | --- | --- |
| `func` |  | The function to summarize within each voxel. |
| `dim` |  | The dimension upon which to summarize (i.e. "z", "intensity", etc.) |
