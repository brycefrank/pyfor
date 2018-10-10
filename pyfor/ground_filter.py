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
    Holds functions and data for implementing Kraus and Pfeifer (1998) ground filter. The Kraus and Pfeifer ground filter
    is a simple filter that uses interpolation of errors and an iteratively constructed surface to filter ground points.
    This filter is used in FUSION software, and the same default values for the parameters are used in this implementation.
    """

    def __init__(self, cloud, cell_size, a=1, b=4, g=-2, w=2.5, iterations=5, cpu_optimize=False):
        """
        :param cloud: The input `Cloud` object.
        :param cell_size: The cell size of the intermediate surface used in filtering in the same units as the input
        cloud. Values from 1 to 6 are used for best performance.
        :param a: A steepness parameter for the interpolating function.
        :param b: A steepness parameter for the interpolating function.
        :param g: The distance from the surface under which all points are given a weight of 1.
        :param w: The window width from g up considered for weighting.
        :param iterations: The number of iterations, i.e. the number of surfaces constructed.
        :param cpu_optimize: If set to True more memory is used but performance is significantly increased
        """
        self.cloud = cloud
        self.cell_size = cell_size
        self.a = a
        self.b = b
        self.g = g
        self.w = w
        self.iterations = iterations
        self.cpu_optimize = cpu_optimize

    def _compute_weights(self, v_i):
        """
        Computes the weights (p_i) for the residuals (v_i).

        :param v_i: A vector of residuals.
        :return: A vector of weights, p_i
        """
        p_i = np.empty(v_i.shape)
        p_i[v_i <= self.g] = 1
        p_i[np.logical_and(v_i > self.g, v_i <= self.g+self.w)] = 1 / (1 + (self.a * (v_i[np.logical_and(v_i > self.g, v_i <= self.g+self.w)] - self.g)**self.b))
        p_i[v_i > self.g+self.w] = 0
        return p_i

    def _filter(self):
        """
        Runs the actual ground filter. Generally used as an internal function that is called by user functions
        (.bem, .classify, .ground_points).
        """
        # TODO a bit memory intensive, the slowest part is finding the points in the original DF that are ground
        # TODO probably some opportunity for numba optimization, but working well enough for now
        grid = self.cloud.grid(self.cell_size)
        self.cloud.data.points['bins_z'] = self.cloud.data.points.groupby(['bins_x', 'bins_y']).cumcount()
        depth = np.max(self.cloud.data.points['bins_z'])
        z = np.zeros((grid.m, grid.n, depth + 1))
        z[:] = np.nan
        z[self.cloud.data.points['bins_y'], self.cloud.data.points['bins_x'], self.cloud.data.points['bins_z']] = self.cloud.data.points['z']
        p_i = np.zeros((grid.m, grid.n, depth+1))
        p_i[~np.isnan(z)] = 1

        if self.cpu_optimize == True:
            ix = np.zeros((grid.m, grid.n, depth + 1))
            ix[self.cloud.data.points['bins_y'], self.cloud.data.points['bins_x'], self.cloud.data.points['bins_z']] = self.cloud.data.points.index.values

        for i in range(self.iterations):
            surface = np.nansum(z * p_i, axis=2) / np.sum(p_i, axis = 2)
            surface = surface.reshape(grid.m,grid.n,1)
            p_i= self._compute_weights(z - surface)
        final_resid = z - surface

        del p_i
        del surface

        if self.cpu_optimize == True:
            ground_bins = (final_resid <= self.g + self.w).nonzero()
            return(self.cloud.data.points.iloc[ix[ground_bins]])
        else:
            ground_bins = (final_resid <= self.g+self.w).nonzero()
            bin_indexer = list(zip(ground_bins[0], ground_bins[1], ground_bins[2]))
            self.cloud.data.points = self.cloud.data.points.set_index(['bins_y', 'bins_x', 'bins_z'])
            return self.cloud.data.points.loc[bin_indexer].reset_index()

    @property
    def ground_points(self):
        """
        Returns a new `Cloud` object that only contains the ground points.
        :return:
        """
        from pyfor.cloud import CloudData, Cloud
        ground = self._filter()
        return Cloud(CloudData(ground, self.cloud.data.header))

    def bem(self, cell_size):
        """
        Retrieve the bare earth model (BEM).
        :return: A `Raster` object that represents the bare earth model.
        """
        ground_cloud = self.ground_points
        return ground_cloud.grid(cell_size).interpolate(np.min, "z")


    def classify(self, ground_int=2):
        """
        Sets the classification of the original input cloud points to ground (default 2 as per las specification). This
        performs the adjustment of the input `Cloud` object **in place**. Only implemented for `.las` files.

        :param ground_int: The integer to set classified points to, the default is 2 in the las specification for ground
        points.
        """

        if self.cloud.extension == '.las':
            self._filter()
            self.cloud.data.points["classification"][self.cloud.data.points['v_i'] <= self.g + self.w] = ground_int
        else:
            print("This is only implemented for .las files.")

    def normalize(self, cell_size):
        """
        Normalizes the original point cloud **in place**. This creates a BEM as an intermediate product, please see
        `.bem()` to return this directly.

        :param cell_size: The cell_size for the intermediate BEM. Values from 1 to 6 are common.
        """
        bem = self.bem(cell_size)
        df = pd.DataFrame(bem.array).stack().rename_axis(['bins_y', 'bins_x']).reset_index(name='val')
        df = self.cloud.data.points.reset_index().merge(df, how="left").set_index('index')
        self.cloud.data.points['z'] = df['z'] - df['val']
