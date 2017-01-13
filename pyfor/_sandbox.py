coordinates = [(x,y) for x in range(11) for y in range(11)]

import numpy as np
print(coordinates)

min_xyz1 = [453192.93, 4735490.24, 900.14]
max_xyz1 = [454192.92, 4736309.9, 1207.65]

def better_grid_constructor(min_xyz, max_xyz, step):
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

print(better_grid_constructor(min_xyz1, max_xyz1, 100))

