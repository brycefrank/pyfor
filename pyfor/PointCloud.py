import laspy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CloudInfo:
    def __init__(self, filename):
        self.filename = filename

        # Reads file from .las
        self.las = laspy.file.File(filename, mode='r')

        # Creates NumPy array of XYZ (Might delete this)
        self.scaled_xyz = np.column_stack(
            (self.las.x, self.las.y, self.las.z))  # Basic positional information for points.

        # Gets extent and creates a grid variable for later.
        self.mins = self.las.header.min
        self.maxes = self.las.header.max
        self.grid = None
        self.grid_step = None

        # Constructs Dataframe from lasfile
        self.dataframe = pd.DataFrame(self.scaled_xyz, columns=['x', 'y', 'z'])
        self.dataframe['cell_id'] = 0
        self.dataframe['classification'] = 1  # Per las documentation, unclassified points are labeled as 1.

    def grid_constructor(self, step):
        ## Should this just occur upon init?

        # Round all mins down.
        min_xyz = self.mins
        max_xyz = self.maxes
        min_xyz = [int(coordinate) for coordinate in min_xyz]

        # Round all maxes up.
        max_xyz = [int(np.ceil(coordinate)) for coordinate in max_xyz]

        # Get each coordinate for grid boundaries list constructor.
        x_min = min_xyz[0]
        x_max = max_xyz[0]
        y_min = min_xyz[1]
        y_max = max_xyz[1]

        # Construct and return list of x,y coordinates of grid.
        self.grid = [(x, y) for x in range(x_min, x_max + step, step) for y in range(y_min, y_max + step, step)]
        self.grid_step = step

    def cell_sort(self):
        # Sorts datapoints into cells of defined size

        def within_square(origin_x, origin_y, step, point):
            '''Determines if point 'i' is within a square of size 'step'''
            ix = point[0]
            iy = point[1]
            if (origin_x <= ix and ix < origin_x + step) and (origin_y <= iy and iy < origin_y + step):
                return True
            else:
                return False

        def fill_dataframe(index, cell_id, dataframe):
            dataframe.set_value(index, 'cell_id', cell_id)

        step = self.grid_step
        dataframe = self.dataframe
        coordinates = 'fixme' #FIXME: retrieve xy coordinate list from dataframe, or use scaled_xyz
        cell_id = 0
        for origin in self.grid:
            point_id = 0
            for coordinate in coordinates:
                if within_square(origin[0], origin[1], step, coordinate) == True:
                    fill_dataframe(point_id, cell_id, dataframe)
                point_id += 1
            cell_id += 1

    def ground_classify(self):
        # FIXME: notation consistency
        df = self.dataframe
        # Construct list of ID's to adjust
        grouped = df.groupby('cell_id')
        ground_id = [df.idxmin()['z'] for key, df in grouped]
        # Adjust to proper classification
        for coordinate in ground_id:
            df.set_value(coordinate, 'classification', 2)
        return df

    def create_BEM(self):
        pass


thing1 = CloudInfo(r"C:\Paco\0_25\000001.S.CLAS.las")


