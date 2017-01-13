# For a given .las file, finds the extent and creates a grid of defined size, which is listed as an array of tuples?

import laspy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def grid_constructor(min_xyz, max_xyz, step):
    # Round all mins down.
    min_xyz = [int(coordinate) for coordinate in min_xyz]

    # Round all maxes up.
    max_xyz = [int(np.ceil(coordinate)) for coordinate in max_xyz]

    # Get each coordinate for grid boundaries list constructor.
    x_min = min_xyz[0]
    x_max = max_xyz[0]
    y_min = min_xyz[1]
    y_max = max_xyz[1]

    # Construct and return list of x,y coordinates of grid.
    # Adding step to each maximum ensures the entire extent of the point cloud is gridded.
    return [(x, y) for x in range(x_min, x_max+step, step) for y in range(y_min, y_max+step, step)]

def within_square(origin_x,origin_y, step, point):
    '''Determines if point 'i' is within a square of size 'step'''
    ix = point[0]
    iy = point[1]
    if (origin_x<=ix and ix<origin_x+step) and (origin_y<=iy and iy<origin_y+step):
        return True
    else:
        return False


class CloudInfo:
    def __init__(self, filename):
        self.filename = filename
        self.las = laspy.file.File(filename, mode='r')
        self.scaled_xyz = np.column_stack((self.las.x, self.las.y, self.las.z)) #Basic positional information for points.
        self.mins = self.las.header.min
        self.maxes = self.las.header.max
    def get_extent(self):
        pass
    def grid_constructor(self):
        pass
    def create_BEM(self):
        pass

def fill_dataframe(index, cell_id, dataframe):
    dataframe.set_value(index, 'cell_id', cell_id)

def thing(grid, coordinates, dataframe):
    '''A very slow way to sort points into a square grid'''
    cell_id = 0
    for origin in grid:
        point_id = 0
        for coordinate in coordinates:
            if within_square(origin[0], origin[1], step1, coordinate) == True:
                fill_dataframe(point_id, cell_id, dataframe)
            point_id += 1
        cell_id += 1
        print(cell_id)

#TODO: begin structuring class

def construct_dataframe(xyz_coordinates):
    '''Constructs standard pandas dataframe with basic xyz positioning.'''
    df = pd.DataFrame(xyz_coordinates, columns = ['x', 'y', 'z'])
    df = ['cell_id'] = 0
    return df



df1.to_pickle(r"C:\pyfor\testpickles\df1.pkl")

df1= pd.read_pickle(r"C:\pyfor\testpickles\df1.pkl")

df1= df1.sample(10000)

df1.plot(kind='scatter', x='x', y='y', c='z', lw=0)



plt.show()