import numpy as np
from scipy.ndimage.morphology import grey_opening
from scipy.interpolate import griddata

def window_size(k):
    b = 2
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

    # TODO decide if this slope works or not
    s = slope(elev_array, w_k, w_k_1)
    s = 1

    if w_k <= 3:
        return(dh_0)
    elif w_k > 3:
        return(s * (w_k - w_k_1) * c + dh_0)
    else:
        return(dh_max)

def zhang(array, number_of_windows, dh_max, dh_0, c, grid, interp_method = "nearest"):
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
    w_k_list = list(map(window_size, range(number_of_windows)))
    w_k_min = w_k_list[0]
    A = array
    m = A.shape[0]
    n = A.shape[1]
    flag = np.zeros((m, n))
    for w_k in enumerate(w_k_list):
        opened = grey_opening(array, (w_k[1], w_k[1]))
        if w_k[1] == w_k_min:
            w_k_1 = 0
        else:
            w_k_1 = w_k_list[w_k[0] - 1]
        for i in range(0, m):
            P_i = A[i,:]
            Z = P_i
            Z_f = opened[i,:]
            dh_t = dht(Z, w_k[1], w_k_1, dh_0, dh_max, c)
            for j in range(0, n):
                if Z[j] - Z_f[j] > dh_t:
                    flag[i, j] = w_k[1]
            P_i = Z_f
            A[i,:] = P_i

    if np.sum(flag) == 0:
        print("No ground points classified.")
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

