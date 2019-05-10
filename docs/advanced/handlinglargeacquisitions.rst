Handling Large Acquisitions
===========================

The pyfor ``collections`` module allows for efficient handling of large acquisitions with paralell
processing of user-define functions. This document presents a couple examples of large acquisition
processing.

Creating Project-Level Canopy Height Models
-------------------------------------------

A ``CloudDataFrame`` is the integral part of managing large point cloud acquisitions. It is initialized
by passing a directory that contains ``.las`` or ``.laz`` files:

.. code-block:: python

    import pyfor
    my_collection = pyfor.collection.from_dir("my_collection_dir", n_jobs=3)

Here we have instantiated a collection such that the processing function will be done in parallel across
three cores.

It will be useful to set the coordinate reference system for the project. This is done via the ``.crs`` attribute:

.. code-block:: python

    import pyproj
    crs = pyproj.proj({'init': 'epsg:26910'}).srs
    my_collection.crs = crs

Here we must grapple with a few things. First, we will want to process buffered tiles to eliminate edge effects from
normalization. This requires defining a few things for the collection.

First, we want to set the collection tiles such
that the output rasters will be consistent for a specified grid cell size. This is done by manipulating the internal
geometries stored in ``my_collection.tiles``. By default, these describe the ``.las/z`` bounding boxes, but we want to
ensure the output rasters are exactly the correct size so that they line up correctly. We can do this easily with the
``.retile_raster`` helper function. This is done **in place** for the collection.

.. code-block:: python

    my_collection.retile_raster(0.5, 500, buffer=20)

The first argument is the desired resolution of the output raster. The second argument is the resolution of the new
tile sets. ``500`` means they will be processed in 500m x 500m chunks. Finally ``buffer=20`` means that each tile will
be buffered by 20 meters, such that we can process a large buffered tile to eliminate edge effects.

Next, we want to define a function to process each buffered tile. This is where the flexibility of the collection
enters in.


.. code-block:: python

    def my_process_func(buffered_cloud, tile):
        buffer_dist = 20
        buffered_cloud.normalize(1)

        # Generate CHM of buffered cloud
        chm = buffered_cloud.chm(0.5, interp_method="nearest", pit_filter="median")

        # Define output bounding box (remove the buffered part)
        coords = list(tile.exterior.coords)
        bbox = coords[0][0] + buffer_dist, coords[2][0] - buffer_dist, coords[0][1] + buffer_dist, coords[1][
            1] - buffer_dist
        chm.force_extent(bbox)

        # Make a readable name for this particular tile
        flat_coords = [int(np.floor(coord)) for coord in bbox]
        tile_str = '{}_{}_{}_{}'.format(*flat_coords)

        # Write out the canopy height model
        chm.write('{}.tif'.format(tile_str))

The above function is lengthy, but really quite simple. First, we normalize and create a canopy height model for some
given buffered point cloud. Then, we restrict its output to remove the buffer using ``.force_extent``. Finally, we write
out the canopy height model to file with a nicely formatted name.

Finally, we can execute the processing job with the following:

.. code-block:: python

    my_collection.par_apply(my_process_func)
