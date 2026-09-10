Benchmarks
==========

How does pyfor compare to other tools that process LiDAR for forest inventory? This page reports a
benchmark of pyfor, `lidR <https://github.com/r-lidar/lidR>`_, and
`PDAL <https://pdal.org>`_ on a single tile, and what the comparison revealed about pyfor.

Every number here comes from ``benchmarks/bench.py``, which is in the repository. Run it again on
your own machine and you should get the same picture, though the absolute times will differ.

What was run
------------

The tile is the one that ships with pyfor, ``pyfortest/data/test.laz``: 217,222 points over a 200 m
by 200 m area, a 728 KiB ``.laz``.

Each tool runs its own idiomatic version of the same six operations:

.. list-table::
    :header-rows: 1

    * - operation
      - pyfor
      - lidR
      - PDAL
    * - ``read``
      - ``Cloud(path)``
      - ``readLAS()``
      - ``readers.las``
    * - ``readwrite``
      - read, then ``Cloud.write()`` as ``.laz``
      - read, then ``writeLAS()``
      - ``readers.las`` then ``writers.las``
    * - ``normalize``
      - ``Cloud.normalize(1)``, the Zhang et al. (2003) progressive morphological filter
      - ``normalize_height(las, tin())``, a Delaunay triangulation of the ground points
      - ``filters.smrf`` then ``filters.hag_nn``
    * - ``chm``
      - ``grid(1).raster("max", "z")`` written to GeoTIFF
      - ``rasterize_canopy(las, 1, p2r())``
      - ``writers.gdal`` with ``output_type: max``
    * - ``chm_normalized``
      - normalize, then the ``chm`` operation
      - ``normalize_height(las, tin())``, then the ``chm`` operation
      - ``filters.smrf``, ``filters.hag_nn``, then ``writers.gdal`` on ``HeightAboveGround``
    * - ``metrics``
      - ``normalize(1)`` then ``grid(20).standard_metrics(2)``, 25 rasters
      - ``normalize_height(las, tin())`` then ``pixel_metrics(stdmetrics, res = 20)``
      - not supported: PDAL has no area based metrics stage

The normalization algorithms are **not** equivalent, and neither are the two metric suites, so those
rows compare complete workflows rather than identical arithmetic. The rasterization rows are
equivalent, and the harness verifies that below.

Method
------

* Every measurement is the wall clock of a fresh process (``Rscript``, ``python``, ``pdal``), so
  package loading is included. The ``in-process`` column is the time from after the interpreter and
  libraries are loaded to the end of the operation, as reported by the tool itself, which is what
  the operation itself costs.
* One warmup run per measurement, discarded, then the median of three timed runs with the page cache
  warm.
* One thread per tool: PDAL's ``readers.las`` ``threads`` option and pyfor's single threaded numpy
  operations. lidR is used without a ``future`` plan, so it runs sequentially.
* PDAL's raster grid is pinned to the grid pyfor produces (``origin_x``, ``origin_y``, ``width``,
  ``height``). Both tools now use the same origin convention, so the pinning states which grid to
  use rather than papering over two different ones; the verification below is what shows the two
  agree cell for cell.

Results
-------

Median of three runs, in seconds:

.. list-table::
    :header-rows: 1

    * - operation
      - tool
      - wall clock
      - in-process
    * - ``read``
      - pyfor
      - 1.644
      - 0.071
    * -
      - lidR
      - 5.251
      - 0.354
    * -
      - pdal
      - 0.219
      - --
    * - ``readwrite``
      - pyfor
      - 1.751
      - 0.140
    * -
      - lidR
      - 5.309
      - 0.440
    * -
      - pdal
      - 0.318
      - --
    * - ``normalize``
      - pyfor
      - 2.297
      - 0.740
    * -
      - lidR
      - 6.351
      - 1.453
    * -
      - pdal
      - 0.654
      - --
    * - ``chm``
      - pyfor
      - 1.723
      - 0.140
    * -
      - lidR
      - 5.506
      - 0.510
    * -
      - pdal
      - 0.239
      - --
    * - ``chm_normalized``
      - pyfor
      - 2.334
      - 0.736
    * -
      - lidR
      - 6.498
      - 1.702
    * -
      - pdal
      - 0.775
      - --
    * - ``metrics``
      - pyfor
      - 3.025
      - 1.409
    * -
      - lidR
      - 7.001
      - 2.025

Reading the numbers:

* On a tile this size the wall clock is dominated by starting the process and loading libraries, not
  by the work. pyfor spends about 1.6 s of its 1.72 s ``chm`` run on ``import pyfor``, which pulls
  in geopandas, rasterio, matplotlib, and laspy. lidR spends about 5.0 s of its 5.51 s loading R and
  the lidR and terra stack. PDAL is a single binary and starts in about 55 ms, which is why its wall
  clock is close to its cost of work.
* Per unit of work, pyfor is the fastest of the three on every operation it implements: about 3.6x
  faster than lidR at rasterizing a CHM (0.140 s against 0.510 s), about 2.0x faster at normalizing
  (0.740 s against 1.453 s), and about 1.4x faster at the metric suite (1.409 s against 2.025 s).
  lidR's metric suite is `stdmetrics`, which computes 56 rasters where pyfor computes 25, and lidR's
  timing includes writing all 56 to a GeoTIFF while pyfor returns 25 rasters in memory, so that row
  is closer than the numbers alone suggest.
* PDAL wins the wall clock everywhere, but its work is also the point of comparison: reading this
  tile costs pyfor 0.071 s of work against PDAL's 0.219 s for the whole process, and writing it
  costs 0.140 s against 0.318 s. Normalizing and rasterizing are a wash between the two, with PDAL
  ahead on the wall clock because it starts so much faster. Note that the normalization algorithms
  differ, so that row compares pyfor's progressive morphological filter against PDAL's SMRF plus
  nearest neighbour height assignment rather than the same arithmetic.

Verification: are the tools computing the same thing?
-----------------------------------------------------

Timings only mean something if the tools did the same work, so the harness compares the CHMs.

On the shipped tile, pyfor and PDAL produce the same canopy height model, cell for cell:

* pyfor against PDAL, chm: 39,696 of 40,000 cells compared, no cell that only one of them fills,
  maximum difference **0.0 m**, identical.
* pyfor against an independent numpy binning of the same points, written to figure out the cells
  without pyfor: identical, maximum difference 0.0 m.
* pyfor against lidR, chm: 39,692 of 40,000 cells compared, 4 cells that only pyfor fills and 15
  that only lidR fills, maximum difference 16.4 m, mean absolute difference 0.024 m.

**On a synthetic tile whose extent sits on the cell lattice**, where points sit in the middle of
cells and no cell maximum depends on a point on the outer edge of the extent, all three agree
exactly: pyfor against lidR 2,499 of 2,500 cells with a maximum difference of 0.0 m, and pyfor
against PDAL the same. (The 2,500th cell is empty in all three.)

What pyfor and PDAL disagree about is only the height of the normalized product: ``chm_normalized``
differs by 0.47 m on average, because pyfor normalizes with the Zhang et al. (2003) filter and PDAL
with SMRF plus nearest neighbour height assignment. Those are different algorithms, not different
implementations of one.

Against lidR the difference that remains is lidR's own choices, which a probe on two cells exposes.
One point sits exactly on the grid line between the two, carrying a height of 99:

   .. code-block:: text

       tool    rows north to south
       pyfor   [ 99.0,  1.0]      the boundary point goes to the north cell
       pdal    [ 99.0,  1.0]      the same
       lidR    [None,  2.0, 99.0] the boundary point goes to the south cell, and the raster grows a
                                  cell to hold a point sitting on the outer edge

pyfor and PDAL follow GDAL's rule, where a point on a grid line belongs to the cell above it. lidR
puts it in the cell below and grows the raster by a cell for points on the far edge, which is why
its raster is 51 by 51 where the other two are 50 by 50 and why a handful of edge cells differ.

What the benchmark changed
--------------------------

The first run of this benchmark found that pyfor's rasters could not be lined up with anyone else's,
and the work that followed is what the numbers above now show:

1. **Rasters are gridded on a snapped grid.** A raster's origin used to be the extent of the data it
   was computed from, so two tiles of one project, or a pyfor raster and a GDAL one, described
   different cells and could not be compared or mosaicked. The origin is now snapped to a multiple
   of the cell size, the target aligned pixels convention that ``gdal_translate -tap``, terra, and
   lidR use. On the shipped tile the old grid started at 405000.01 and the snapped one starts at
   405000.0, which changes the maximum of 731 of the 39,691 cells the two have in common, by
   0.023 m on average and 17.19 m in the worst cell. `results.json` reports this as
   ``grid_convention_effect``.
2. **Points on a cell boundary follow GDAL.** A point exactly on a horizontal grid line belonged to
   the cell below it; it now belongs to the cell above it, as it does in GDAL, rasterio, and PDAL.
   4,285 of the tile's 217,222 points sit on such a line.
3. **A grid can be stated explicitly.** :class:`.GridSpec` is an origin, a cell size, and a size in
   cells, and :meth:`.Cloud.grid`, :meth:`.Cloud.chm`, :meth:`.Cloud.normalize`, and both ground
   filters accept one. :meth:`.CloudDataFrame.grid_spec` builds one that covers a whole collection,
   and :meth:`.Retiler.retile_raster` snaps its tiling origin to the cell size.
4. **Trimming a raster refuses to guess.** :meth:`.Raster.force_extent` used to round a bounding box
   to the nearest cell, which silently labelled an array with a grid it was not on. It now raises
   when the box does not fall on the cells.
5. **Still open: startup cost.** ``import pyfor`` costs about 1.5 s, most of the wall clock for a
   short script. Importing rasterio, geopandas, and matplotlib lazily, where they are actually used,
   would fix that.

Reproducing this
----------------

.. code-block:: bash

    # pyfor, from the repository
    pip install -e ".[test]"

    # lidR: archived on CRAN since 2026-06-09 because it depends on the archived rlas,
    # so install the last release from the archive
    Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/lidR/lidR_4.3.2.tar.gz", repos = NULL, type = "source")'

    # PDAL: any 2.6+ build with the smrf, hag, and gdal plugins
    conda install -c conda-forge pdal

    python benchmarks/bench.py --pdal "$(which pdal)" --runs 3

``benchmarks/bench.py --help`` lists the options. Results land in ``benchmarks/results/results.json``,
including the exact command line and the effective PDAL pipeline behind every number.

The measurements on this page were taken on 2026-09-09 with pyfor 0.4.0, laspy 2.7.0, numpy 2.5.3,
R 4.6.0, lidR 4.3.2, rlas 1.9.5, and PDAL 2.10.2, on an Intel Core i5-1345U (12 threads, 16 GB).
``benchmarks/results/results.json`` records the environment of the run that is reported here.
