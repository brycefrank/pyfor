import numpy as np
import pandas as pd

class GroundFilter:
    pass

class Zhang2003:
    """
    Implements Zhang et. al (2003), a progressive morphological ground filter. This filter uses an opening operation
    combined with progressively larger filtering windows to remove features that are 'too steep'. This particular
    implementation interacts only with a raster, so the output resolution will be dictated by the `cell_size` argument.
    """
    def __init__(self, cloud, cell_size, n_windows=5, dh_max=2, dh_0=1, b=2, interp_method="nearest"):
        """
        :param n_windows: The number of windows to construct for filtering.
        :param dh_max: The maximum height threshold.
        :param dh_0: The starting null height threshold.
        :param cell_size: The cell_size used to construct the array for filtering, also the output size of the BEM.
        :param interp_method: The interpolation method used to fill nan values in the final BEM.
        """
        self.cloud = cloud
        self.n_windows = n_windows
        self.dh_max = dh_max
        self.dh_0 = dh_0
        self.b = b
        self.cell_size = cell_size
        self.interp_method = interp_method

        self.grid = self.cloud.grid(self.cell_size)
        self.array = self.grid.raster(np.min, "z").array

    def _window_size(self, k, b):
        return(2 * k * b + 1)

    def _dhmax(self, elev_array):
        """
        Calculates the maximum height difference for an elevation array.

        :param elev_array:
        :return:
        """
        return(np.max(elev_array) - np.min(elev_array))


    def _slope(self, elev_array, w_k, w_k_1):
        """
        Calculates the slope coefficient.

        Returns the slope coefficient s for a given elev_aray and w_k
        """

        return(self._dhmax(elev_array) / ((w_k - w_k_1) / 2))


    def _dht(self, elev_array, w_k, w_k_1, dh_0, dh_max, c):
        """"
        Calculates dh_t.

        :param elev_array: A 1D array of elevation values
        :param w_k: An integer representing the window size
        :param w_k_1: An integer representing the previous window size
        """
        #s = self._slope(elev_array, w_k, w_k_1)
        s = 1

        if w_k <= 3:
            return(dh_0)
        elif w_k > 3:
            return(s * (w_k - w_k_1) * c + dh_0)
        else:
            return(dh_max)

    def _filter(self):
        from scipy.ndimage.morphology import grey_opening

        w_k_list = [self._window_size(i, self.b) for i in range(self.n_windows)]
        w_k_min = w_k_list[0]
        A = self.array
        m = A.shape[0]
        n = A.shape[1]
        flag = np.zeros((m, n))
        for k, w_k in enumerate(w_k_list):
            opened = grey_opening(self.array, (w_k, w_k))
            if w_k == w_k_min:
                w_k_1 = 0
            else:
                w_k_1 = w_k_list[k - 1]
            for i in range(0, m):
                P_i = A[i,:]
                Z = P_i
                Z_f = opened[i,:]
                dh_t = self._dht(Z, w_k, w_k_1, self.dh_0, self.dh_max, self.cell_size)
                for j in range(0, n):
                    if Z[j] - Z_f[j] > dh_t:
                        flag[i, j] = w_k
                P_i = Z_f
                A[i,:] = P_i

        if np.sum(flag) == 0:
            return(None)

        # Remove interpolated cells
        empty = self.grid.empty_cells
        empty_y, empty_x = empty[:,0], empty[:,1]
        A[empty_y, empty_x] = np.nan
        B = np.where(flag != 0, A, np.nan)
        return B

    def bem(self):
        from scipy.interpolate import griddata
        from pyfor.rasterizer import Raster
        B = self._filter()

        # Interpolate on our newly found ground cells
        X, Y = np.mgrid[0:self.grid.m, 0:self.grid.n]
        C = np.where(np.isfinite(B) == True)
        vals = B[C[0], C[1]]
        dem_array = griddata(np.stack((C[0], C[1]), axis = 1), vals, (X, Y), method=self.interp_method)

        return(Raster(dem_array, self.grid))

class KrausPfeifer1998:
    """
    Holds functions and data for implementing Kraus and Pfeifer (1998) ground filter. The Kraus and Pfeifer ground filter
    is a simple filter that uses interpolation of errors and an iteratively constructed surface to filter ground points.
    This filter is used in FUSION software, and the same default values for the parameters are used in this implementation.
    """

    def __init__(self, cloud, cell_size, a=1, b=4, g=-2, w=2.5, iterations=5, tolerance=0):
        """
        :param cloud: The input `Cloud` object.
        :param cell_size: The cell size of the intermediate surface used in filtering in the same units as the input
        cloud. Values from 1 to 40 are common, depending on the units in which the original point cloud is projected.
        :param a: A steepness parameter for the interpolating function.
        :param b: A steepness parameter for the interpolating function.
        :param g: The distance from the surface under which all points are given a weight of 1.
        :param w: The window width from g up considered for weighting.
        :param iterations: The number of iterations, i.e. the number of surfaces constructed.
        """
        self.cloud = cloud
        self.cell_size = cell_size
        self.a = a
        self.b = b
        self.g = g
        self.w = w
        self.iterations = iterations

        if tolerance == 0:
            self.tolerance = self.g + self.w

    def _compute_weights(self, v_i):
        """
        Computes the weights (p_i) for the residuals (v_i).

        :param v_i: A vector of residuals.
        :return: A vector of weights, p_i
        """
        p_i = np.empty(v_i.shape)
        p_i[v_i <= self.g] = 1
        middle = np.logical_and(v_i > self.g, v_i <= self.g+self.w)
        p_i[middle] = 1 / (1 + (self.a * (v_i[middle] - self.g)**self.b))
        p_i[v_i > self.g+self.w] = 0
        return p_i

    def _filter(self):
        """
        Runs the actual ground filter. Generally used as an internal function that is called by user functions
        (.bem, .classify, .ground_points).
        """
        np.seterr(divide='ignore', invalid='ignore')

        # TODO probably some opportunity for numba optimization, but working well enough for now
        grid = self.cloud.grid(self.cell_size)
        self.cloud.data.points['bins_z'] = self.cloud.data.points.groupby(['bins_x', 'bins_y']).cumcount()
        depth = np.max(self.cloud.data.points['bins_z'])
        z = np.zeros((grid.m, grid.n, depth + 1))
        z[:] = np.nan
        z[self.cloud.data.points['bins_y'], self.cloud.data.points['bins_x'], self.cloud.data.points['bins_z']] = self.cloud.data.points['z']
        p_i = np.zeros((grid.m, grid.n, depth+1))
        p_i[~np.isnan(z)] = 1

        for i in range(self.iterations):
            surface = np.nansum(z * p_i, axis=2) / np.sum(p_i, axis = 2)
            surface = surface.reshape(grid.m,grid.n,1)
            p_i= self._compute_weights(z - surface)
        final_resid = z - surface

        del p_i
        del surface

        ix = np.zeros((grid.m, grid.n, depth + 1))
        ix[self.cloud.data.points['bins_y'], self.cloud.data.points['bins_x'],
           self.cloud.data.points['bins_z']] = self.cloud.data.points.index.values
        ground_bins = (final_resid <= self.g + self.w).nonzero()
        self.cloud.data.points = self.cloud.data.points.reset_index()
        return self.cloud.data.points.loc[ix[ground_bins]]


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

        :param cell_size: The cell size of the BEM, this is independent of the cell size used in the intermediate \
        surfaces.
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
        # Rebin the cloud to the new cell size
        # TODO make this into a standalone function (in raster, grid?), it is used in several other places
        self.cloud.grid(cell_size)
        self.cloud.data._update()
        df = pd.DataFrame(bem.array).stack().rename_axis(['bins_y', 'bins_x']).reset_index(name='val')
        df = self.cloud.data.points.reset_index().merge(df, how="left").set_index('index')
        self.cloud.data.points['z'] = df['z'] - df['val']
