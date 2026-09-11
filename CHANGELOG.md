# Unreleased

## Documentation

The user manual moved from Sphinx to Astro Starlight and is published by the new `docs` workflow to
GitHub Pages at https://brycefrank.com/pyfor.

- Every narrative page was ported to Markdown under `docs/src/content/docs/`, and the images
  moved with them.
- The API reference is generated from the docstrings in `pyfor/` by
  `docs/scripts/generate_api.py`, which reads the package statically with `griffe` and writes one
  page per module. Anchor names keep their dotted Sphinx form, so existing deep links resolve.
- URLs from the old site, such as `/pyfor/html/topics/normalization.html`, redirect to their new
  pages.

# 0.4.0

Modernization release, September 9, 2026.

pyfor now runs on Python 3.10 and newer against current versions of its dependencies, and the point
cloud core is numpy rather than pandas. `laspy` 1.x, Python 2.7/3.7, LASTools, and `numba` are no
longer supported.

## Breaking Changes

1. `laspy` 2.x is required. All reading and writing was ported to the laspy 2 API, the `header` of a
   `CloudData` object is a `laspy.LasHeader`, and `Cloud` accepts a `laspy.LasData` object directly.
2. **`CloudData.points` is a structured numpy array**, not a pandas `DataFrame`. The dimension names
   are unchanged (`x`, `y`, `z`, `intensity`, `return_num`, `classification`, `flag_byte`,
   `scan_angle_rank`, `user_data`, `pt_src_id`, plus `red`, `green`, and `blue` where the point
   format has them) and are listed by `points.dtype.names`. Dataframe-only access is gone:
   `.columns`, `.head()`, `.sample()`, `.iloc`, `.loc`, and `.values` must become `dtype.names`,
   positional slicing, `points["name"]`, and plain numpy indexing.
3. Bins are no longer columns of the points. `Grid` and `VoxelGrid` hold their own `bins_x`,
   `bins_y` (and `bins_z`), `cell_ids`, and `n_cells`, so binning a cloud no longer modifies the
   point data.
4. `.lax` spatial indexing was removed. `CloudDataFrame.create_index`, the `CloudDataFrame.indexed`
   property, the `indexed` argument of `par_apply`, and `CloudDataFrame.map_poly` are gone, along
   with the `laxpy` dependency and the LASTools `lasindex` binary. Polygon queries now read points in
   chunks (`pyfor.cloud.read_polygon`) and keep only the points inside the query polygon in memory.
5. `Grid.cells`, the pandas `GroupBy`, was replaced by explicit numpy reductions: `Grid.reduce`,
   `Grid.cell_values`, `Grid.cell_counts`, `Grid.cell_ranks`, `Grid.cell_percentiles`,
   `Grid.percentile_raster`, `Grid.expand`, and `Grid.n_cells`.
6. `Grid.metrics` always returns `(dimension, function)` multiindex columns. It previously returned
   flat columns when every dimension was given a single function.
7. **Rasters are computed on a snapped grid.** A raster's origin used to be the extent of the data it
   was computed from, so two tiles of one project, or a pyfor raster and a GDAL one, described
   different cells and could not be compared or mosaicked. The origin is now snapped to a multiple
   of the cell size, the target aligned pixels convention that `gdal_translate -tap`, terra and lidR
   use, and a point lying exactly on a horizontal grid line now belongs to the cell above it, as it
   does in GDAL, rasterio and PDAL. On the test tile this changes the maximum of 731 of 39,691 cells,
   by 0.023 m on average and 17.19 m in the worst cell, so heights and metrics per cell move
   slightly. `Raster.force_extent` now raises when a bounding box does not fall on the cells of the
   raster instead of quietly rounding it, which used to leave an array labelled with a grid it was
   not on.
8. `ImportedGrid.array` is the raster as read from the file, north up. `Cloud.subtract` used to flip
   it before looking values up.
9. Removed `pyfor.ground_filter.GroundFilter` (an empty placeholder class),
   `pyfor.rasterizer.Raster.from_rasterio` (an unimplemented stub that took no arguments), and
   `ImportedGrid.in_raster` (replaced by `ImportedGrid.array`, and the file handle is now closed
   after reading).
10. `pyfor.rasterizer.sample_array` takes `(array, bins_x, bins_y)` and `metrics.summarize_return_num`
   takes an array rather than a `Series`.
11. `numba` is no longer a dependency. Clipping (`pyfor.clip.ray_trace`) is vectorized numpy and
   returns identical results.
12. Packaging moved to `pyproject.toml`. `setup.py`, `setup.cfg`, `MANIFEST`, and
    `test_environment.yml` were removed, and `.travis.yml` was replaced by a GitHub Actions workflow.

## Changes to Codebase

### Core Data Model

1. Points are built in one place, `pyfor.cloud.points_from_columns`, so every `CloudData.points`
   array has the same layout: one field per dimension, addressable by name.
2. `points_from_laspy` converts laspy dimensions and point records to that layout, and
   `Cloud.read_polygon` builds it from a chunked read of a `.las`/`.laz` file.
3. `pyfor.cloud`, `pyfor.clip`, `pyfor.ground_filter`, and `pyfor.voxelizer` no longer import pandas.
   Pandas remains for tabular outputs only: `Grid.metrics`, the `standard_metrics*` summaries, and
   the `CloudDataFrame` of tile geometries.
4. `Cloud.write` dispatches to `LASData.write` or `PLYData.write` as before. `LASData.write` copies
   the header, sizes it to the points held in memory, and calls `LasData.update_header()` so written
   bounds and point counts always match the data.
5. `Cloud.subtract`, `Cloud.clip`, `Cloud.filter`, and `Cloud.convex_hull` operate on arrays
   directly. `Cloud.normalize` updates the cloud's bounds after modifying heights.
6. `Cloud.crs` is initialized from the coordinate reference system in the file header, so rasters
   written from a georeferenced las file carry their CRS instead of a warning. It was `None` until
   a user set it by hand.

### Cell Reductions

1. Added `pyfor.rasterizer.reduce_cells`, which reduces a value per point for every occupied cell.
   `count`/`size`, `sum`, `mean`, `min`, `max`, `std`, and `var` are computed with
   `np.bincount`/`np.*.at` in a single pass, and any other callable is applied to the values of each
   occupied cell in turn.
2. Added `pyfor.rasterizer.percentile_cells`, a vectorized equivalent of calling
   `np.percentile` on the values of every cell. Heights percentiles are the most expensive reduction
   pyfor performs: the standard metric suite on a 1 m grid over a 200 m tile went from 22.1 s to
   0.81 s.
3. `Grid.cell_ranks` replaces the `groupby(...).cumcount()` used by the Kraus and Pfeifer filter.
4. `Grid.reduce` and `Grid.interpolate` accept a boolean mask, so a reduction can be restricted to a
   subset of the points while keeping the grid of the whole cloud. This is how the bare earth models
   of both ground filters are now built.

### Rasterizer

1. Added `GridSpec`, the origin, cell size, and size in cells of a grid, with
   `GridSpec.covering(min_x, min_y, max_x, max_y, cell_size)` for the snapped grid covering an
   extent. `Grid`, and therefore `Cloud.grid`, `Cloud.chm`, `Cloud.normalize`, and both ground
   filters, accept one through their `spec` keyword. A spec that does not cover the cloud it is
   handed is refused rather than silently dropping points.
2. `Grid.interpolate` queried the interpolation backend on a grid offset by one cell, which shifted
   interpolated rasters (canopy height models) by one cell. The query grid is now aligned with the
   cell bins: interpolated cells reproduce un-interpolated cell values exactly, and the previous
   behavior was off by up to 18.9 m on the test tile.
3. `ImportedGrid` bins points using the raster transform instead of `np.linspace`, which drifted
   from the raster grid, and reads the raster with a context manager.
4. `Grid`, `VoxelGrid`, and `ImportedGrid` clip bin indices to the array bounds, so a point on the
   outermost boundary can no longer index past the end of an array. `Grid` and `VoxelGrid` also keep
   at least one cell per dimension: a cloud narrower than the cell size used to produce a zero
   column or row grid, which crashed on the first reduction.
5. `np.int` was replaced with `np.int64`.

### Ground Filter

1. `KrausPfeifer1998` decides the ground points with a per point mask (`_ground_mask`) instead of
   mapping three dimensional indices back to rows of a dataframe. `_filter` still returns the
   filtered points.
2. `KrausPfeifer1998.bem` grids the bare earth model on the parent cloud at the requested cell size
   and interpolates only the ground points. Previously the model was interpolated on the ground
   points' own grid and then sampled with the parent cloud's bins, which shifted the surface by up
   to one cell whenever the ground returns did not span the extent of the cloud.
3. `Zhang2003.bem(classified=True)` reduces the parent cloud grid to the points classified as ground
   (2), which makes `Cloud.normalize(classified=True)` use the classified model. Previously the
   classified model was computed and discarded, so the option did nothing.
4. `scipy.ndimage.morphology.grey_opening` moved to `scipy.ndimage.grey_opening`, and
   `KrausPfeifer1998._filter` uses `np.errstate` instead of modifying global numpy error state.

### Metrics

1. All grid metrics are computed with the cell reductions described above. `return_num` and
   `total_returns` are `cell_counts` over a mask, and the percentile metrics use
   `Grid.percentile_raster`.
2. `standard_metrics_cloud` works on a structured array, and `canopy_relief_ratio` no longer emits
   divide warnings for cells with a single return.
3. `z_mean_sq` squared its raster with `^` (bitwise xor) instead of `**`. Fixed.
4. `np.alen` was replaced with `len`.

### Collection

1. `_construct_tile_indexed` and `_construct_tile_no_index` were replaced by a single
   `_construct_tile`, which reads the points of intersecting files with `read_polygon` and joins
   them with `numpy.concatenate`. The `args` keyword argument is now passed to the applying function
   in both cases; previously it was dropped when `indexed=True`.
2. File listing is sorted, so tile order is deterministic.
3. `_get_bounding_box` and `_get_datetime` read headers only, without loading points. Datetimes are
   `pandas.Timestamp` values.
4. `CloudDataFrame.grid_spec(cell_size)` builds a :class:`.GridSpec` covering the collection, to
   hand to the processing of every tile so their rasters line up.
5. `Retiler.retile_raster` snaps the tiling origin to the cell size, so tile boundaries fall on the
   same lattice the rasters are gridded on.
6. `CloudDataFrame.crs` is the geopandas CRS of the bounding box geometries. The eager
   `crs = None` assignment that raised on frames without an active geometry column was removed.
7. `CloudDataFrame.plot_metrics` imported a function name that did not exist; it now calls
   `standard_metrics_cloud`.

### Benchmarks

1. Added `benchmarks/bench.py`, which times pyfor, lidR, and PDAL on the same tile for the same six
   operations, pins every tool to the same raster grid, and checks that the tools produce the same
   CHM before reporting times. `benchmarks/lidr.R`, `benchmarks/pyfor_ops.py`, and one PDAL pipeline
   per operation are the per tool halves.
2. `docs/src/content/docs/topics/benchmarks.md` reports the results, and the two problems the
   first run of the benchmark found, both of which are fixed in this release (see the breaking
   changes above):
   * rasters could not be lined up with any other tool, because the grid was anchored at the extent
     of the data rather than snapped to the cell size;
   * a point exactly on a horizontal cell boundary was assigned to a different cell than GDAL and
     PDAL assign it to.
   `benchmarks/bench.py` measures what that was worth, and reports it in `results.json` as
   `grid_convention_effect`.
3. Verified by the harness, so that the timings are comparisons of equivalent work:
   * pyfor and PDAL produce identical CHMs on the test tile, 39,696 cells compared with a maximum
     difference of 0.0 m, and pyfor matches an independent numpy binning of the same points;
   * on a synthetic tile whose extent sits on the cell lattice, pyfor, lidR, and PDAL all produce
     identical CHMs, 2,499 cells with a maximum difference of 0.0 m;
   * against lidR, what remains is lidR's own conventions: a point on a grid line goes to the cell
     below it and the raster grows a cell for points on the far edge, which changes 4 cells that
     only pyfor fills and 15 that only lidR fills on the test tile.

### Packaging & Testing

1. Dependencies moved to `pyproject.toml`, the version is read from `pyfor.__version__`, and the
   `plot` optional extra holds `pyqtgraph`/`PyOpenGL` for `Cloud.plot3d`.
2. Tests were ported to laspy 2 and numpy points, and are run with pytest. `.laz` coverage,
   previously disabled, is enabled now that `lazrs` is a dependency.
3. Removed `test_pcs_exists`, which asserted that the test file's own directory exists, and the
   unused `make_test_collection` helper.
4. The suite runs about twice as fast as the pandas implementation it replaced (12.7 s to 6.6 s), and
   the buffered tile pipeline produces byte identical GeoTIFFs before and after the change.

## Fixed Issues

Both issues from the 0.3.6 era listed here in earlier drafts of this release are fixed, and verified:

1. `Cloud.normalize(classified=True)` now differs from `Cloud.normalize()` on the test tile (mean
   absolute difference of 0.33 m) where it was previously identical.
2. `KrausPfeifer1998.normalize` no longer samples a bare earth model with bins from another grid. The
   bins carried by the raster's grid now agree exactly with the raster's affine transformation, and
   the height error that a one cell origin mismatch used to cause (mean 0.24 m, maximum 16.1 m) can
   no longer occur.

# 0.3.6

Updates between September 9, 2019 and December 1, 2019.

## Changes to Codebase

### Cloud
1. Added `.from_pdal` class method which converts a `PDAL` `python.filter` `ins` argument to a Cloud object.

### Collection
1. `._build_polygons` is now multithreaded.
2. `.create_index` is now multithreaded. Closes #65
3. `from_dir` now supports glob strings. Closes #66
4. `par_apply`  accepts optional keyword arguments using the `args` 
parameter. These are passed to the applying function.
5. Added date-time parsing to collections.

### GISExport
1. Removed some deprecated functions that were only used for tree segmentation

### Ground Filter
1. Fixed a bug where `KrausPfeifer1998.classify` was throwing a key error. Closes #62.

### Metrics
1. Added `all_returns` metric, that counts the number of returns in a cell.

### Rasterizer
1. Gridding of points has been simplified. Closes #67.

### Testing Suite
1. Modified `.travis.yml` for Windows testing environment.
2. Added additional checks for number of `.lax` files produced and length of tile change on `retile_raster` for
collection testing. Removed printing statements.

## Installation
1. `0.3.6` now officially supports conda installation! Dancing in the streets. Check the README for instructions.

## Other
1. Version numbers must now be set in both `setup.py` and `__init__.py` for compatibility with conda-forge

# 0.3.5

Updates between April 20, 2019 and September 9, 2019

A few small maintenance updates and implementation of `metrics`.

## Cloud
1. `.normalize` wrapper now allows for using already classified ground points with the `classified` argument.

## Metrics
1. Added metrics computations for `grid` objects
2. Finalized the standard set of metrics for both `cloud` and `grid` objects via `standard_metrics_grid` and 
`standard_metrics_cloud`

## Raster
1. `.watershed_seg` and `.local_maxima` were removed. Please see [treeseg](https://github.com/brycefrank/treeseg) for 
new implementations of these methods.

## Testing Suite
1. Added tests for `Cloud.subtract`, `Raster.force_extent`.

# 0.3.4 - Hotfixes

Hotixes applied directly to `master` between April 20, 2019 and 0.4.0 release.

## Raster

1. Fixed bug in `force_extent` when appending empty arrays to bottom or top dimensions (May 3, 2019)

## Collection

1. Restructured `_construct_tile`

# 0.3.4

Updates between February 10, 2019 and April 20, 2019. 

In addition to the new features below, the documentation and user manual
have been unified into one website located [here](http://brycefrank.com/pyfor/html/index.html). The `pyfor_manual` 
repository will be deprecated and deleted in the coming months. All of its content has been transferred to the new
website.

## Raster
1. Added `force_extent` function that allows users to force a specific output bounding box for a raster.

## CloudDataFrame
1. Revamped `par_apply` to take advantage of `.lax` files if they are present.
2. Added a `.crs` attribute to set a collection level coordinate reference system.

## Retile
1. Added a class (`collection.Retiler`) and wrapper functions (`CloudDataFrame.retile_raster`, etc) to assist in setting
tile extents for a particular collection. This allows for flexibility when outputting summary rasters and reduced clutter
in the `CloudDataFrame` class.

## Other
1. Removed `plotly` dependencies and functions. These were not essential to the package, and created a lot of dependencies
for the conda installation that are no longer present.
2. Removed `detection` module. This module was poorly maintained and implemented an unoptimized tree detection function.
In the coming months I plan to start implementing detection tasks again, but in a more optimized way.

## Testing Suite
1. Added tests for new `par_apply`
2. Testing for commonly loaded `.las` and `.laz` fields.

# 0.3.3

Updates between December 4th, 2018 and February 10, 2019. `0.3.3` implements a few structural changes and bug fixes.
Its release will be followed by a major restructuring to implement in-memory `.laz` support in 0.3.4.

## Cloud
1. Added warning to `.clip` when no points are present after the clip. #38
2. Minor restructuring to accommodate for a file write bug, addresses #40

## Collection
1. Parallelized `CloudDataFrame.clip`

## CrownSegments
1. Fixed a bug where crown segments were misprojected. #37

## Ground Filter
1. For both `Zhang2003` and `KrausPfeifer1998`, changed filter instantiation to reflect `scikit-learn` type instantiation.

## Other
1. Implemented single sourcing for package versioning. #43
2. Minor adjustments to `environment.yml` to ensure Travis success. #41

# 0.3.2

Updates between October 25th, 2018 and December 4th, 2018. Note: some of these were applied directly to `master` before the merging of this branch.
0.3.2 represents many bug fixes and the addition of a few (relatively performant, your mileage may very) functions to `collection`.
A shoutout to Ben Weinstein, whos diligent bug reporting has made `pyfor` a friendlier package during this update.

## Cloud
1. Fixed a bug with the `name` attribute that returned the entire directory instead of just the filename (without extension)
2. Added `.subtract` function, this allows a user to provide their own (properly referenced) DEM for use in normalizing the parent cloud object.
3. `Cloud.clip` now resets the index of the points dataframe `Cloud.data.points`
4. Resolved issues instantiating `Cloud` objects using `.laz` files.
5. Changed the default normalization algorithm back to `Zhang2003` now that it is working properly again (see below).
6. `Cloud.convex_hull` now returns a single `shapely.geometry.Polygon` instead of `geopandas.GeoSeries`

## Collection
1. Added `bounding_box` attribute that retrieves the bounding box of the entire collection, used in retiling.
2. Added `retile` function to split large acquisitions into smaller tile sizes, for now this just splits into quadrants.
3. Added `clip` function to make memory-optimized spatial queries of collections, for example: clipping a collection of field
plots.
4. Made `CloudDataFrame.index_las` and `CloudDataFrame.from_dir` into internal functions: `CloudDataFrame._index_las` and `CloudDataFrame._from_dir` respectively.

## *Data
1. Added a check for empty dataframe before writing to file.
2. Improved the structure of `LASData` to prevent writing non-existant columns to file.

## Ground Filter
1. Added `normalize` function to `Zhang2003`.
2. Fixed an issue where `KrausPfeifer1998` was producing non-sensical normalizations.

## Rasterizer
1. `Grid` now computes bins starting from the top left of an input `Cloud`. This fixes a lot of unnecessary array flipping further downstream.
A UserWarning was added and will remain in effect until `0.3.3`.

## General Adjustments
1. Now testing multiple file types, `.ply`, `.laz` and `.las`
2. Added `lastools` and `laxpy` to the dependency stack.
3. Edited ~60% of the docstrings to reflect recent changes. Added cross referenced links and other small improvements.
4. Updated the samples to reflect recent changes. Added a few sections to the Normalization and Collections samples.

# 0.3.1

Updates between October 11th, 2018 and October 25th, 2018

## Detection
1. Refactored `LayerStacking` to `Ayrey2017` to cohere more with the citation format.

## Documentation
1. Updated docstrings with 0.3.0 changes.

## Rasterizer
1. Removed `Grid.normalize`, a deprecated function
2. Added a `UserWarning` in the case of undefined coordinate reference system of the Cloud object when writing a `Raster`

## Testing Suite
1. Added many tests to achieve 90% coverage

# 0.3.0

Updates between September 5, 2018 and October 10th, 2018

## Collection
1. Added 'CloudDataFrame', an inherited class from GeoDataFrame used to manage large LiDAR acquisitions.

## Samples
1. Added `Collections` sample
2. Adjusted normalization/bem/etc sample with new updates.

## Windows Compatibility
1. Addressed issues with plotting on Windows 10 - an up-to-date version of PyCharm should work well.

## Cloud
1. Added support for `.ply` files
2. Changed default normalization algorithm to `ground_filter.KrausPfeifer1998` while I debug and restructure `ground_filter.zhang`
3. Made `CloudData` a base class for the new `LASDAta` and `PLYData` classes.

## Rasterizer
1. Changed `Grid.raster` to allow for keyword arguments for passed functions.
2. Added `DeprecationWarning` to `rasterizer.Grid.normalize`, will be replaced with standalone ground filters in 3.1.
3. Added `DetectedTops` object, used for visualizing detected tops from CHM.

## Filter
1. Moved `filter` to `ground_filter`

### KrausPfeif

## Ground_Filter
1. Restructued filters into their own Classes, each with `.bem`, `.classify()` and `.normalize`. Fits better with the structure of the package.
2. Added Kraus and Pfeifer (1998) ground filter after having some issues with `zhang`. This filter is a much simpler ground filter but provides reasonable results.

## environment.yml
1. Enforcing `rasterio > 1.0` requirement which thereby requires use of `conda-forge` channel.
2. Added `plyfile` requirement (see above)

# 0.2.3

Updates between August 5, 2018 and September 5, 2018. These updates are minor improvements
to set up for 0.3.0 release.

## Documentation

1. Moved documentation from ReadTheDocs to brycefrank.com/pyfor
2. Updated documentation main page and internal structure

## Testing Suite

1. Fixed broken clip polygons

## Samples

1. Fixed clip sample (from above)
2. Added LayerStacking sample

## Detection
1. Several improvements to `detection.LayerStacking`

## GISExport
1. Added a project indices function, mostly for internal use.

# 0.2.2

Updates between May 9, 2018 and August 5, 2018.

## Cloud
1. Fixed pandas SettingWithCopyWarning after clip + plot, still needs to be tested for performance (i.e. is this copy
    necessary?)
2. Added plotting for custom dimensions for `Cloud.plot3d()`
3. Moved `pyqtgraph` import statements witin `Cloud.plot3d()` to improve import perormance
4. Added functionality for plotting detected trees. Very rough but functional.
5. Added summary functionality, use `print(some_cloud_object)` to view.

## Rasterizer
1. Watershed segmentation output was oriented incorrectly, fixed. (Actually was fixed via master, putting here for
    reference).
2. Fixed a bug that produced the wrong axes tick mark labels after modification of the Cloud object.
3. Reworked the behavior of `rasterizer.Raster.local_maxima`
    - By default only produces one pixel per detected top, whereas before it was possible to produce many pixels per
    top. This occurred if the detected top pixels were all equal in height.
    - If you prefer this type of behavior, you can set the argument `multi_top` to True
    - The other major rework here is that the function now returns a properly geo-referenced Raster, instead of a raw
    array. This is much more useful w/r/t I/O. 

## Detection
1. Added dedicated detection module
2. Added early version of LayerStacking (Ayrey et al. 2017)
    - This gets as far as the "Overlap map" in their paper
    - Sample forthcoming
    
## Testing Suite
1. Added an second feature to the testing shapefile.
2. Adjusted testing suite to NEON data set for simplicity.
3. Test fixes for the above changes.

## Environment
1. Enforcing rasterio version >= 1.0.2 in the environment for use of MemoryFiles (involved with LayerStacking)

# 0.2.1

Update merged to master: May 9th, 2018

This update was meant to implement features and bug fixes on the tile processing capabilities as well as improve some of the visualization functions. More samples and documentation were added, along with ~90% code coverage. Although the 0.3.0-specific `voxelizer.py` was added, it is not officially supported in this release.

## Samples
1. Added a watershed segmentation sample.
2. Added a clipping sample.
3. Improved normalization sample
4. Added grid metrics sample with the new `as_raster` argument (see below)

## Voxelizer
1. Added basic `VoxelGrid` class with a 3D version of raster

## Rasterizer
### Raster
1. Added a plot argument to the `rasterizer.Raster.watershed_seg()` function. This will plot the segment polygons over the raster object.
2. Fixed a bug in `rasterizer.Raster.iplot3d` that prevented plotting
3. Added `rasterizer.Raster._convex_hull_mask`, this helps plot interpolated rasters correctly by setting values outside
   of the convex hull to nan. This will help with future plotting and writing function to be written.
   columns that describe which dimension and which metric were calculated for the raster in that row. More details can
   be seen in the `Grid_Metrics` sample.
## Grid
1. Removed the plot function from `rasterizer.Grid`, did not seem to fit with the philosophy of the object.
2. Added `as_raster` argument to `rasterizer.Grid.metrics`, this returns a pandas dataframe of the Raster objects that has

## Cloud
1. Added `cloud.Cloud.convex_hull`, returns the convex hull of the two dimensional plane.
2. Reduced `cloud.Cloud.clip` to handle only shapely polygons for maintainability.

## clip_funcs
1. `clip_funcs.poly_clip` now returns the indices of the original input cloud object. This is a cleaner approach than previously implemented.
2. Some slight adjustments to the other functions to accommodate (1).

## General Maintenance
1. More tests were added.
2. Added a sample polygon shapefile: pyfortest/data/clip.shp, this will be used to test the clipping function.
3. Moved some import statements to their respective functions to reduce import time of the package.
4. Changed the theme of the documentation to something a bit more readable.
