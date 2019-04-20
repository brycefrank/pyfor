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
