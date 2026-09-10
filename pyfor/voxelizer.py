import numpy as np
from pyfor.rasterizer import reduce_cells


class VoxelGrid:
    """A 3 dimensional grid representation of a point cloud. This is analagous to the rasterizer.Grid class, but
    with three axes instead of two. VoxelGrids are generally used to produce VoxelRaster objects."""

    def __init__(self, cloud, cell_size):
        self.cell_size = cell_size
        self.cloud = cloud

        min_x, max_x = self.cloud.data.min[0], self.cloud.data.max[0]
        min_y, max_y = self.cloud.data.min[1], self.cloud.data.max[1]
        min_z, max_z = self.cloud.data.min[2], self.cloud.data.max[2]

        self.m = max(1, int(np.floor((max_y - min_y) / cell_size)))
        self.n = max(1, int(np.floor((max_x - min_x) / cell_size)))
        self.p = max(1, int(np.floor((max_z - min_z) / cell_size)))

        points = self.cloud.data.points

        # Create bins
        y_edges = np.linspace(min_y, max_y, self.m)
        bins_x = np.searchsorted(
            np.linspace(min_x, max_x, self.n), points["x"]
        )
        bins_y = (
            np.searchsorted(
                -y_edges,
                -points["y"],
                side="right",
                sorter=(-y_edges).argsort(),
            )
            - 1
        )
        bins_z = np.searchsorted(np.linspace(min_z, max_z, self.p), points["z"])

        # The outermost edge of each search space is inclusive, so a coordinate on the boundary of the
        # cloud can bin one past the last voxel of an axis.
        self.bins_x = np.clip(bins_x, 0, self.n - 1)
        self.bins_y = np.clip(bins_y, 0, self.m - 1)
        self.bins_z = np.clip(bins_z, 0, self.p - 1)

        self.cell_ids = (
            (self.bins_x * self.m + self.bins_y) * self.p + self.bins_z
        )

    @property
    def n_cells(self):
        """:return: The total number of voxels, occupied or not."""
        return self.m * self.n * self.p

    def voxel_raster(self, func, dim):
        """Creates a 3 dimensional voxel raster, analagous to rasterizer.Grid.raster.

        :param func: The function to summarize within each voxel.
        :param dim: The dimension upon which to summarize (i.e. "z", "intensity", etc.)
        """
        voxel_grid = np.zeros(self.n_cells)
        cells, values = reduce_cells(
            self.cloud.data.points[dim], self.cell_ids, self.n_cells, func
        )
        voxel_grid[cells] = values

        return voxel_grid.reshape(self.m, self.n, self.p)
