# For a given .las file, finds the extent and creates a grid of defined size, which is listed as an array of tuples?

# import laspy
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# def grid_constructor(min_xyz, max_xyz, step):
#     # Round all mins down.
#     min_xyz = [int(coordinate) for coordinate in min_xyz]
#
#     # Round all maxes up.
#     max_xyz = [int(np.ceil(coordinate)) for coordinate in max_xyz]
#
#     # Get each coordinate for grid boundaries list constructor.
#     x_min = min_xyz[0]
#     x_max = max_xyz[0]
#     y_min = min_xyz[1]
#     y_max = max_xyz[1]
#
#     # Construct and return list of x,y coordinates of grid.
#     return [(x, y) for x in range(x_min, x_max+step, step) for y in range(y_min, y_max+step, step)]

# def within_square(origin_x,origin_y, step, point):
#     '''Determines if point 'i' is within a square of size 'step'''
#     ix = point[0]
#     iy = point[1]
#     if (origin_x<=ix and ix<origin_x+step) and (origin_y<=iy and iy<origin_y+step):
#         return True
#     else:
#         return False


# def fill_dataframe(index, cell_id, dataframe):
#     dataframe.set_value(index, 'cell_id', cell_id)
#
# def point_assign(grid, coordinates, dataframe, step):
#     '''A very slow way to sort points into a square grid'''
#     cell_id = 0
#     for origin in grid:
#         point_id = 0
#         for coordinate in coordinates:
#             if within_square(origin[0], origin[1], step, coordinate) == True:
#                 fill_dataframe(point_id, cell_id, dataframe)
#             point_id += 1
#         cell_id += 1
#         print(cell_id)

# def construct_dataframe(xyz_coordinates):
#     '''Constructs standard pandas dataframe with basic xyz positioning.'''
#     df = pd.DataFrame(xyz_coordinates, columns = ['x', 'y', 'z'])
#     df['cell_id'] = 0
#     df['classification'] = 1 # Per las documentation, unclassified points are labeled as 1.
#     return df

def sample_graph(dataframe):
    dataframe = dataframe.sample(10000)
    dataframe.plot(kind='scatter', x='x', y='y', c='cell_id', lw=0, cmap="flag", s=2)
    plt.show()


def construct_bem(dfgroup):
    pass

#TODO: re-structure this process, make a whole function?
# thing1 = CloudInfo(r"C:\Paco\0_25\000001.S.CLAS.las")

# df1 = construct_dataframe(thing1.scaled_xyz)
#
# thing(grid_constructor(thing1.mins, thing1.maxes, 100), thing1.scaled_xy, df1, 100)
#
#
#
# df1.to_pickle(r"C:\pyfor\testpickles\df2.pkl")

df1= pd.read_pickle(r"C:\pyfor\testpickles\df1.pkl")
df1['classification']=1
print(df1.head())

def ground_classify(df):
    # Construct list of ID's to adjust
    grouped = df.groupby('cell_id')
    ground_id = [df.idxmin()['z'] for key,df in grouped]
    # Adjust to proper classification
    for coordinate in ground_id:
        df.set_value(coordinate, 'classification', 2)
    return df

df1 = ground_classify(df1)


grouped = df1.groupby('classification')

# df1 = df1.sample(10000)
df1.plot(kind='scatter', x='x', y='y', c='classification', lw=0, s=1)
plt.show()
# grouped = df1.groupby('cell_id') # splits into multiple dataframes
#
# for key,df in grouped:
#     # print(df.max()['z'] - df.min()['z'])
#     print(df.idxmin()['z'])

thing1 = CloudInfo(r"C:\Paco\0_25\000001.S.CLAS.las")
print(thing1.maxes, thing1.mins)