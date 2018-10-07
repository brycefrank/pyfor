import numpy as np
import pandas as pd
from scipy.ndimage.morphology import grey_opening
from scipy.interpolate import griddata

def window_size(k, b):
    return(2 * k * b + 1)

def dhmax(elev_array):
    """
    Calculates the maximum height difference for an elevation array.

    :param elev_array:
    :return:
    """
    return(np.max(elev_array) - np.min(elev_array))


def slope(elev_array, w_k, w_k_1):
    """
    Calculates the slope coefficient.

    Returns the slope coefficient s for a given elev_aray and w_k
    """
    return(dhmax(elev_array) / ((w_k - w_k_1) / 2))


def dht(elev_array, w_k, w_k_1, dh_0, dh_max, c):
    """"
    Calculates dh_t.

    :param elev_array: A 1D array of elevation values
    :param w_k: An integer representing the window size
    :param w_k_1: An integer representing the previous window size
    """
    s = slope(elev_array, w_k, w_k_1)
    s = 1

    if w_k <= 3:
        return(dh_0)
    elif w_k > 3:
        return(s * (w_k - w_k_1) * c + dh_0)
    else:
        return(dh_max)

def zhang(array, number_of_windows, dh_max, dh_0, c, grid, b = 2, interp_method = "nearest"):
    """
    Implements Zhang et. al (2003), a progressive morphological ground filter. This returns a matrix of Z values for
    each grid cell that have been determined to be actual ground cells.

    :param array: The array to interpolate on, usually an aggregate of the minimum Z value
    #TODO fix this to be max window size
    :param number_of_windows:
    :param dh_max: The maximum height threshold
    :param dh_0: The starting null height threshold
    :param c: The cell size used to construct the array
    :param grid: The grid object used to construct the array
    :return: An array corresponding to the filtered points, can be used to construct a DEM via the Raster class
    """

    w_k_list = [window_size(i, b) for i in range(number_of_windows)]
    w_k_min = w_k_list[0]
    A = array
    m = A.shape[0]
    n = A.shape[1]
    flag = np.zeros((m, n))
    for k, w_k in enumerate(w_k_list):
        opened = grey_opening(array, (w_k, w_k))
        if w_k == w_k_min:
            w_k_1 = 0
        else:
            w_k_1 = w_k_list[k - 1]
        for i in range(0, m):
            P_i = A[i,:]
            Z = P_i
            Z_f = opened[i,:]
            dh_t = dht(Z, w_k, w_k_1, dh_0, dh_max, c)
            for j in range(0, n):
                if Z[j] - Z_f[j] > dh_t:
                    flag[i, j] = w_k
            P_i = Z_f
            A[i,:] = P_i

    if np.sum(flag) == 0:
        return(None)

    # Remove interpolated cells
    empty = grid.empty_cells
    empty_y, empty_x = empty[:,0], empty[:,1]
    A[empty_y, empty_x] = np.nan
    B = np.where(flag != 0, A, np.nan)

    # Interpolate on our newly found ground cells
    X, Y = np.mgrid[0:grid.m, 0:grid.n]
    C = np.where(np.isfinite(B) == True)
    vals = B[C[0], C[1]]
    dem_array = griddata(np.stack((C[0], C[1]), axis = 1), vals, (X, Y), method=interp_method)

    return(dem_array)

class KrausPfeifer1998:
    """
    Holds functions and data for implementing Kraus and Pfeifer (1998) ground filter.
    """

    def __init__(self, cloud, cell_size, a=1, b=4, g=-2, w=2.5, iterations=5):
        self.cloud = cloud
        self.cell_size = cell_size
        self.a = a
        self.b = b
        self.g = g
        self.w = w
        self.iterations = iterations

    def _compute_weights(self, v_i):
        p_i = np.empty(v_i.shape)
        p_i[v_i <= self.g] = 1
        p_i[np.logical_and(v_i > self.g, v_i <= self.g+self.w)] = 1 / (1 + (self.a * (v_i[np.logical_and(v_i > self.g, v_i <= self.g+self.w)] - self.g)**self.b))
        p_i[v_i > self.g+self.w] = 0
        return p_i

    def _filter(self):
        """
        Runs the actual ground filter. Generally used as an internal function that is called by user functions (.bem, .classify, .ground_points)
        :return:
        """
        import pandas as pd
        from pyfor.cloud import CloudData, Cloud
        import inspect
        grid = self.cloud.grid(self.cell_size)
        surface = grid.raster(np.mean, "z")
        for i in range(self.iterations):
            # Find the values of surface above/below points
            df = pd.DataFrame(surface.array).stack().rename_axis(['bins_y', 'bins_x']).reset_index(name='val')

            # Append the values of surface above/below each point
            surface_vec = self.cloud.data.points.reset_index().merge(df, how = "left").set_index('index')['val']

            # Compute the residuals and plot
            self.cloud.data.points['v_i'] = self.cloud.data.points['z'] - surface_vec
            self.cloud.data.points['p_i'] = self._compute_weights(self.cloud.data.points['v_i'])
            #print(self.cloud.data.points['z'].iloc[0], self.cloud.data.points['v_i'].iloc[0], self.cloud.data.points['p_i'][0])

            # Weight the original z values and construct new surface
            weighted_cells = grid.cells.apply(lambda cell: np.average(cell['z'], weights = cell['p_i'])).reset_index()
            array = np.full((grid.m, grid.n), np.nan)
            array[weighted_cells["bins_y"], weighted_cells["bins_x"]] = weighted_cells[0]
            surface.array = array

        # If _filter was called by bem, return the final surface
        # TODO This caller stuff is functional but feels awkward. Probably best to break apart this function
        caller = inspect.stack()[1][3]
        if caller == 'bem':
            return surface
        elif caller == 'classify':
            self.cloud.data.points[self.cloud.data.points['v_i'] <= self.g+self.w] = 2
        elif caller == 'ground_points':
            ground = self.cloud.data.points[self.cloud.data.points['v_i'] <= self.g + self.w]
            return Cloud(CloudData(ground, self.cloud.data.header))
        elif caller == 'normalize':
            # TODO rework cloud.normalize to take a Raster as input and output the normalized cloud?
            df = pd.DataFrame(surface.array).stack().rename_axis(['bins_y', 'bins_x']).reset_index(name='val')
            self.cloud.data.points = self.cloud.data.points.reset_index().merge(df, how = "left").set_index('index')
            self.cloud.data.points['z'] = self.cloud.data.points['z'] - self.cloud.data.points['val']
            self.cloud.normalized = True

    @property
    def bem(self):
        return self._filter()

    def classify(self, ground_int=2):
        """
        Sets the classification of the original input cloud points to 'ground'. Not yet implemented.
        :return:
        """
        self._filter()

    @property
    def ground_points(self):
        return self._filter()

    def normalize(self):
        """
        Normalizes the original point cloud in place
        :return:
        """
        self._filter()
