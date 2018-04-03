# Functions for rasterizing
import numpy as np
import pandas as pd

class Grid:
    """The grid object constructs a grid from a given Cloud object
    and cell_size and contains functions useful for manipulating
    rasterized data."""
    def __init__(self, cloud, cell_size):
        """
        Sorts the point cloud into a gridded form such that every point in the las file is assigned a cell coordinate
        with a resolution equal to cell_size

        :param cell_size: The size of the cell for sorting
        :param indices: The indices of self.points to plot
        :return: Returns a dataframe with sorted x and y with associated bins in a new columns
        """
        self.las = cloud.las

        # TODO Need to update headers when new cloud is constructed
        min_x, max_x = self.las.header.min[0], self.las.header.max[0]
        min_y, max_y = self.las.header.min[1], self.las.header.max[1]

        self.m = int(np.floor((max_y - min_y) / cell_size))
        self.n = int(np.floor((max_x - min_x) / cell_size))

        # Create bins
        bins_x = np.searchsorted(np.linspace(min_x, max_x, self.n), self.las.x)
        bins_y = np.searchsorted(np.linspace(min_y, max_y, self.m), self.las.y)

        # Add bins and las data to a new dataframe
        df = pd.DataFrame({'x': self.las.x, 'y': self.las.y, 'z': self.las.z,
                           'bins_x': bins_x, 'bins_y': bins_y})
        self.data = df

    def boolean_summary(self, func, dim):
        """
        Calculates a column in self.data that is a boolean of whether
        or not that point is the point that corresponds to the function passed.

        For example, this can be used to create a boolean mask of points that
        are the minimum z point in their respective cell.

        :param func: The function to calculate on each group.
        :param dim: The dimension of the point cloud as a string (x, y or z)
        """

        mask = self.data.groupby(['bins_x', 'bins_y'])[dim].transform(func) == self.data[dim]
        return(mask)

    def get_missing_cells(self):
        """
        TODO docstring
        """

        # Determine which cells have at least one point in them
        mask = self.boolean_summary(np.min, 'z')
        self.data['mask'] = mask
        non_empty_cells = self.data.loc[self.data['mask']]

        # Sort by cell IDs
        grouped_sort = non_empty_cells.sort_values(['bins_x', 'bins_y'])

        # Initialize an array container
        arr = np.empty((0, 2))

        for x_bin in range(1, self.m):
            # Subset the dataframe for each value of x_bin
            x_col = grouped_sort.loc[grouped_sort['bins_x'] == x_bin]

            # Construct a list to feed to missing_elements and get the missing elements
            L = list(x_col['bins_y'])

            if len(L) > 0:
                start, end = L[0], self.m
                missing = sorted(set(range(start, end + 1)).difference(L))

                # Create numpy array of the missing x_bin and y_bins
                ones = np.full(len(missing), x_bin)
                vals = np.array(missing)
                stacked = np.stack((ones, vals), axis = 1)

                # Append to the container array
                arr = np.append(arr, stacked, axis = 0)
        return(arr)
