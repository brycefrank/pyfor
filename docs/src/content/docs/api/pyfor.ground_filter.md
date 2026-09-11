---
title: pyfor.ground_filter
slug: api/pyfor.ground_filter
description: API reference for the pyfor.ground_filter module.
sidebar:
  order: 5
---

## Classes

<a id="pyfor.ground_filter.Zhang2003"></a>

## Zhang2003

```python
Zhang2003(cell_size, n_windows = 5, dh_max = 2, dh_0 = 1, b = 2, interp_method = 'nearest')
```

Implements Zhang et. al (2003), a progressive morphological ground filter. This filter uses an opening operation combined with progressively larger filtering windows to remove features that are 'too steep'. This particular implementation interacts only with a raster, so the output resolution will be dictated by the `cell_size` argument.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `n_windows` |  |  |
| `dh_max` |  |  |
| `dh_0` |  |  |
| `b` |  |  |
| `cell_size` |  |  |
| `interp_method` |  |  |

### Methods

<a id="pyfor.ground_filter.Zhang2003.bem"></a>

#### bem

```python
bem(cloud, classified = False, spec = None)
```

Retrieve the bare earth model (BEM). Unlike `KrausPfeifer1998`, the cell size is defined upon initialization of the filter, and thus it is not required to retrieve the bare earth model from the filter.

| Parameter | Type | Description |
| --- | --- | --- |
| `cloud` |  | A Cloud object. |
| `classified` |  | If True, the bare earth model is constructed from the points classified as ground (2). |
| `spec` |  | An optional `GridSpec` for the model, see `Cloud.grid`. |

**Returns:** A `Raster` object that represents the bare earth model.

<a id="pyfor.ground_filter.Zhang2003.normalize"></a>

#### normalize

```python
normalize(cloud, classified = False, spec = None)
```

Normalizes the original point cloud **in place**. This creates a BEM as an intermediate product, please see `.bem()` to return this directly.

| Parameter | Type | Description |
| --- | --- | --- |
| `cloud` |  | The input cloud object to normalize. |
| `classified` |  | If True, the bare earth model is constructed from the points classified as ground (2). |
| `spec` |  | An optional `GridSpec` for the bare earth model, see `Cloud.grid`. |

<a id="pyfor.ground_filter.KrausPfeifer1998"></a>

## KrausPfeifer1998

```python
KrausPfeifer1998(cell_size, a = 1, b = 4, g = -2, w = 2.5, iterations = 5, tolerance = 0)
```

Holds functions and data for implementing Kraus and Pfeifer (1998) ground filter. The Kraus and Pfeifer ground filter is a simple filter that uses interpolation of errors and an iteratively constructed surface to filter ground points. This filter is used in FUSION software, and the same default values for the parameters are used in this implementation.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `cell_size` |  |  |
| `a` |  |  |
| `b` |  |  |
| `g` |  |  |
| `w` |  |  |
| `iterations` |  |  |
| `tolerance` |  |  |

### Methods

<a id="pyfor.ground_filter.KrausPfeifer1998.ground_points"></a>

#### ground_points

```python
ground_points(cloud)
```

Returns a new `Cloud` object that only contains the ground points.

**Returns:** 

<a id="pyfor.ground_filter.KrausPfeifer1998.bem"></a>

#### bem

```python
bem(cloud, cell_size, spec = None)
```

Retrieve the bare earth model (BEM).

| Parameter | Type | Description |
| --- | --- | --- |
| `cloud` |  | A cloud object. |
| `cell_size` |  | The cell size of the BEM, this is independent of the cell size used in the intermediate surfaces. |
| `spec` |  | An optional `GridSpec` for the model, see `Cloud.grid`. |

**Returns:** A `Raster` object that represents the bare earth model.

<a id="pyfor.ground_filter.KrausPfeifer1998.classify"></a>

#### classify

```python
classify(cloud, ground_int = 2)
```

Sets the classification of the original input cloud points to ground (default 2 as per las specification). This performs the adjustment of the input `Cloud` object **in place**. Only implemented for `.las` files. Additionally, the original ground classification is preserved if it exists, so this will only add additional ground points for already classified point clouds.

| Parameter | Type | Description |
| --- | --- | --- |
| `cloud` |  | A cloud object. |
| `ground_int` |  | The integer to set classified points to, the default is 2 in the las specification for ground points. |

<a id="pyfor.ground_filter.KrausPfeifer1998.normalize"></a>

#### normalize

```python
normalize(pc, cell_size, spec = None)
```

Normalizes the original point cloud **in place**. This creates a BEM as an intermediate product, please see `.bem()` to return this directly.

| Parameter | Type | Description |
| --- | --- | --- |
| `pc` |  | A cloud object. |
| `cell_size` |  | The cell_size for the intermediate BEM. Values from 1 to 6 are common. |
| `spec` |  | An optional `GridSpec` for the model, see `Cloud.grid`. |
