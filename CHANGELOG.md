# 0.3.0

Updates between September 5, 2018 and ...

## Collection

1. Added 'CloudDataFrame', an inherited class from GeoDataFrame used to manage large LiDAR acquisitions.
    - Ability to do geopositional indexing of LAS tiles. Much of this class is internal and used to

## Samples
1. Added `Collections` sample

## Readme
1. Reduced README length

## Windows Compatibility
1. Addressed issues with plotting on Windows 10 - an up-to-date version of PyCharm should work well.

# Parser
1. Added a new module, parser, that is mostly used for internal functions that attempt to parse the CRS for a given las file.

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
