## Samples
1. [In Progress] Added a watershed segmentation sample.
2. [In Progress] Added a clipping sample.
3. Improved normalization sample
4. Improved grid metrics sample with the new as_raster argument (see below)

## Voxelizer
1. Added basic VoxelGrid class

## Rasterizer
1. Added a plot argument to the rasterizer.Raster.watershed_seg() function. This will plot the segment polygons over the raster object.
2. Removed the plot function from rasterizer.Grid, did not seem to fit with the philosophy of the object.
3. Fixed a bug in rasterizer.Raster.iplot3d that prevented plotting
4. Added rasterizer.Raster._convex_hull_mask, this helps plot interpolated rasters correctly by setting values outside
   of the convex hull to nan. This will help with future plotting and writing function to be written.
5. Added as_raster argument to rasterizer.Grid.metrics, this returns a pandas dataframe of the Raster objects that has
   columns that describe which dimension and which metric were calculated for the raster in that row. More details can
   be seen in the Grid_Metrics sample.

## Cloud
1. Added cloud.Cloud.convex_hull, returns the convex hull of the two dimensional plane.
2. Reduced cloud.Cloud.clip to handle only shapely polygons for maintainability.

## clip_funcs
1. clip_funcs.poly_clip now returns the indices of the original input cloud object. This is a cleaner approach than previously implemented.
2. Some slight adjustments to the other functions to accommodate (1).

## General Maintenance
1. More tests were added.
	- Cloud.clip
	- plotting methods
2. Added a sample polygon shapefile: pyfortest/data/clip.shp, this will be used to test the clipping function.
3. Moved some import statements to their respective functions to reduce import time of the package.
