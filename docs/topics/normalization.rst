Normalization
=============

One of the most integral parts of LiDAR analysis is determining which points represent
the ground. Once this step is completed, we can construct bare earth models (BEMs) and
normalize our point clouds to produce reliable estimates of height, canopy height models
and other products.

`pyfor` offers a few avenues for normalization, ground filtering and the creation of bare earth
models. All of these methods are covered in the advanced Ground Filtering document. Here, only
the convenience wrapper is covered.

The convenience wrapper `Cloud.normalize` is a function that filters the cloud object for ground
points, creates a bare earth model, and uses this bare earth model to normalize the object in
place. That is, **it conducts the entire normalization process from the raw data and does not
leverage existing classified ground points**. See the advanced Ground Filtering document to use
existing classified ground points.

It uses the `Zhang2003` filter by default and takes as its first argument the resolution
of the bare earth model:

.. code-block:: python

    import pyfor
    tile = pyfor.cloud.Cloud('my_tile.las')
    tile.normalize(1)

