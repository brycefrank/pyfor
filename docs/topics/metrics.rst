Area-Based Metrics
=============

The area-based approach (ABA) is ubiquitous in modern forest inventory systems. `pyfor` enables the
computation of a set of area-based metrics for point clouds for individual point clouds (`Cloud` objects)
and for gridded point clouds (`Grid` objects).

The following block demonstrates a minimal example of creating standard metrics for a gridded tile.

.. code-block:: python

    import pyfor
    tile = pyfor.cloud.Cloud('my_tile.las')
    tile.normalize(1)
	
	grid = tile.grid(20)
	std_metrics = grid.standard_metrics(2, )
