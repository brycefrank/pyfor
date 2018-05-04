## 0.3.0

This branch is committed toward developing efficient handling of large acquisitions of data. This workflow is slightly different in its approach than the interactive workflow introduced in 0.2.0, and requires careful handling of in-memory operations. Things are still taking shape, but expect at least some integration of `joblib` and `dask`. Both of these packages interact well with multiprocessing, and `dask` integrates well with clusters.

A dedicated changelog is located [here](https://github.com/brycefrank/pyfor/blob/0.3.0/changelog_0.3.0.md).

0.3.0 is intended to be the final pre-release, and is the last major piece of the package framework. A period of testing and minor updates will follow. Following this, 1.0 will be released, which will provide a stable platform for rolling release updates.

Other coming attractions:
  - Multiprocessing samples / Cloud computing samples
  - Voxel visualizations (voxels are already implemented in 0.2.1)
  - More tree detection algorithms
  - Visualizations of large acquisitions
