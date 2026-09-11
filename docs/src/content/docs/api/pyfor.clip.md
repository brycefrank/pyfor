---
title: pyfor.clip
slug: api/pyfor.clip
description: API reference for the pyfor.clip module.
sidebar:
  order: 1
---

## Functions

<a id="pyfor.clip.square_clip"></a>

### square_clip

```python
square_clip(points, bounds)
```

Clips a square from a tuple describing the position of the square.

| Parameter | Type | Description |
| --- | --- | --- |
| `points` |  | A structured numpy array of points, as held in `CloudData.points`. |
| `bounds` |  | A tuple of length 4, min y and max y coordinates of the square. |

**Returns:** A boolean mask, true is within the square, false is outside of the square.

<a id="pyfor.clip.ray_trace"></a>

### ray_trace

```python
ray_trace(x, y, poly)
```

Determines for some set of x and y coordinates, which of those coordinates is within `poly`. Ray trace is generally called as an internal function, see `poly_clip`

| Parameter | Type | Description |
| --- | --- | --- |
| `x` |  | A 1D numpy array of x coordinates. |
| `y` |  | A 1D numpy array of y coordinates. |
| `poly` |  | The coordinates of a polygon as a numpy array (i.e. from geo_json['coordinates'] |

**Returns:** A 1D boolean numpy array, true values are those points that are within `poly`.

<a id="pyfor.clip.poly_clip"></a>

### poly_clip

```python
poly_clip(points, poly)
```

Returns the indices of `points` that are within a given polygon. This differs from `ray_trace` in that it enforces a small "pre-clip" optimization by first clipping to the polygon bounding box. This function is directly called by `Cloud.clip`.

| Parameter | Type | Description |
| --- | --- | --- |
| `points` |  | A structured numpy array of points, as held in `CloudData.points`. |
| `poly` |  | A shapely Polygon, with coordinates in the same CRS as the point cloud. |

**Returns:** A 1D numpy array of indices corresponding to points within the given polygon.
