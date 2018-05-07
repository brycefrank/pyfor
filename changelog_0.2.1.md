## Samples
1. [In Progress] Added a watershed segmentation sample.
2. [In Progress] Added a clipping sample.

## Voxelizer
1. Added basic VoxelGrid class

## Rasterizer
1. Added a plot argument to the rasterizer.Raster.watershed_seg() function. This will plot the segment polygons over the raster object.
2. Removed the plot function from rasterizer.Grid, did not seem to fit with the philosophy of the object.

## Cloud
1. Added cloud.Cloud.convex_hull, returns the convex hull of the two dimensional plane.
2. Reduced cloud.Cloud.clip to handle only shapely polygons for maintainability.

## clip_funcs
1. clip_funcs.poly_clip now returns the indices of the original input cloud object. This is a cleaner approach than previously implemented.
2. Some slight adjustments to the other functions to accommodate (1).

## General Maintenance
1. More tests were added [in progress].
	- Cloud.clip
	- plotting methods
2. Added a sample polygon shapefile: pyfortest/data/clip.shp, this will be used to test the clipping function.
3. Moved some import statements to their respective functions to reduce import time of the package.
