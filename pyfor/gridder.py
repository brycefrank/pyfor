# Playing with the idea of a grid object

class Grid:
    """
    Contains a gridded CloudData frame and functions associated with gridded data
    """
    def __init__(self, cloud, cell_size):
        self.cloud = cloud
        self.cell_size = cell_size
        # TODO Need to update headers when new cloud is constructed
        min_x, max_x = self.cloud.las.header.min[0], self.cloud.las.header.max[0]
        min_y, max_y = self.cloud.las.header.min[1], self.cloud.las.header.max[1]

        m = int(np.floor((max_y - min_y) / cell_size) + 1)
        n = int(np.floor((max_x - min_x) / cell_size) + 1)

        # Create bins
        bins_x = np.searchsorted(np.linspace(min_x, max_x, n + 1), self.cloud.las.x)
        bins_y = np.searchsorted(np.linspace(min_y, max_y, m + 1), self.cloud.las.y)

        # Add bins and las data to a new dataframe
        df = pd.DataFrame({'x': self.cloud.las.x, 'y': self.cloud.las.y, 'z': self.cloud.las.z,
                           'bins_x': bins_x, 'bins_y': bins_y})

        # Add the grid attribute to the parent cloud
        self.cloud.grid = df
