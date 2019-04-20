Clipping
========

Often times we want to clip out LiDAR points using a shapefile. This can be done using pyfor's
Cloud.clip method. pyfor integrates with geopandas and shapely, convenient geospatial packages
for Python, to provide a way to clip point clouds.

.. code-block:: python

    import pyfor
    import geopandas

    # Load point cloud
    pc = pyfor.cloud.Cloud("../data/test.las")

As input to the clipping function we need any `shapely.geometry.Polygon` our heart desires, as
long as its coordinates correspond to the same physical space as the `Cloud` object. Here I extract
a `Polygon` from a shapefile using `geopandas`:

.. code-block:: python

    # Load point cloud
    polys = geopandas.read_file("../data/clip.shp")
    poly = poly_frame["geometry"].iloc[0]

Finally, pass the `Polygon` to the clipping function. This function returns a new `Cloud` object.

.. code-block:: python

    # Load point cloud
    clipped = pc.clip(poly)
