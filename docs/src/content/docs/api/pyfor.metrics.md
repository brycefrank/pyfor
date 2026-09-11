---
title: pyfor.metrics
slug: api/pyfor.metrics
description: API reference for the pyfor.metrics module.
sidebar:
  order: 6
---

## Functions

<a id="pyfor.metrics.summarize_return_num"></a>

### summarize_return_num

```python
summarize_return_num(return_nums)
```

Gets the number of returns by return number.

| Parameter | Type | Description |
| --- | --- | --- |
| `return_nums` |  | A `numpy.ndarray` of the return number of each point. |

**Returns:** A `pandas.Series` of return number counts by return number.

<a id="pyfor.metrics.summarize_percentiles"></a>

### summarize_percentiles

```python
summarize_percentiles(z, pct = all_pct)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `z` |  | A `numpy.ndarray` of z values. |

<a id="pyfor.metrics.pct_above_heightbreak"></a>

### pct_above_heightbreak

```python
pct_above_heightbreak(grid, r = 0, heightbreak = 'mean')
```

Calculates the percentage of first returns above the mean. This needs its own function because it summarizes multiple columns of the point cloud, and is therefore more complex than typical summarizations (i.e. percentiles). This returns a `pyfor.rasterizer.Raster` object.

| Parameter | Type | Description |
| --- | --- | --- |
| `grid` |  | A `pyfor.rasterizer.Grid` object |
| `r` |  | The return number to constrain to. Must be a positive integer. If r=0, all points will be considered (this is the default behavior). |
| `heightbreak` |  | The height at which to summarize. If a number is given, this will be interpreted as the height at which points will be considered "above". If the string "mean" is given (this is the default), will use the mean height of that cell, for example, to construct the "pct_above_mean" metric. |

<a id="pyfor.metrics.grid_percentile"></a>

### grid_percentile

```python
grid_percentile(grid, percentile)
```

Calculates a percentile raster.

| Parameter | Type | Description |
| --- | --- | --- |
| `percentile` |  | The percentile (a number between 0 and 100) to compute. |

<a id="pyfor.metrics.z_max"></a>

### z_max

```python
z_max(grid)
```

Calculates maximum z value.

<a id="pyfor.metrics.z_min"></a>

### z_min

```python
z_min(grid)
```

Calculates minimum z value.

<a id="pyfor.metrics.z_std"></a>

### z_std

```python
z_std(grid)
```

Calculates standard deviation of z value.

<a id="pyfor.metrics.z_var"></a>

### z_var

```python
z_var(grid)
```

Calculates variance of z value.

<a id="pyfor.metrics.z_mean"></a>

### z_mean

```python
z_mean(grid)
```

Calculates mean of z value.

<a id="pyfor.metrics.z_iqr"></a>

### z_iqr

```python
z_iqr(grid)
```

Calculates interquartile range of z value.

<a id="pyfor.metrics.vol_cov"></a>

### vol_cov

```python
vol_cov(grid, r, heightbreak)
```

Calculates the volume covariate (percentage first returns above two meters times mean z)

<a id="pyfor.metrics.z_mean_sq"></a>

### z_mean_sq

```python
z_mean_sq(grid)
```

Calculates the square of the mean z value.

<a id="pyfor.metrics.canopy_relief_ratio"></a>

### canopy_relief_ratio

```python
canopy_relief_ratio(grid, mean_z_arr, min_z_arr, max_z_arr)
```

<a id="pyfor.metrics.return_num"></a>

### return_num

```python
return_num(grid, num)
```

Compute the number of returns that match `num` for a grid object

<a id="pyfor.metrics.all_returns"></a>

### all_returns

```python
all_returns(grid)
```

<a id="pyfor.metrics.total_returns"></a>

### total_returns

```python
total_returns(grid)
```

<a id="pyfor.metrics.standard_metrics_grid"></a>

### standard_metrics_grid

```python
standard_metrics_grid(grid, heightbreak)
```

<a id="pyfor.metrics.standard_metrics_cloud"></a>

### standard_metrics_cloud

```python
standard_metrics_cloud(points, heightbreak)
```
