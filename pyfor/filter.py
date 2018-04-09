import numpy as np
from numba import jit

# PARAMETERS
# TODO put this in function calls later

def window_size(k):
    b = 2
    return(2 * k * b + 1)


def dhmax(elev_array):
    return(np.max(elev_array) - np.min(elev_array))


def slope(elev_array, w_k, w_k_1):
    """
    Returns the slope coefficient s for a given elev_aray and w_k
    """
    return(dhmax(elev_array) / ((w_k - w_k_1) / 2))


def dht(elev_array, w_k, w_k_1, dh_0, dh_max, c):
    """"
    :param elev_array: A 1D array of elevation values
    :param w_k: An integer representing the window size
    :param w_k_1: An integer representing the previous window size
    """

    s = slope(elev_array, w_k, w_k_1)

    if w_k <= 3:
        return(dh_0)
    elif w_k > 3:
        return(s * (w_k - w_k_1) * c + dh_0)
    else:
        return(dh_max)

def erosion(Z, w_k, n):
    Z_f = []
    for j in range(1, n+1):
        lb = np.floor(j - (w_k) / 2)
        ub = np.ceil(j + (w_k) / 2)
        a = [Z[l] for l in range(int(lb), int(ub)) if l >= 0 and l <= len(Z) - 1]
        Z_f.append(np.min(a))
    return(np.asarray(Z_f))

def dilation(Z, w_k, n):
    Z_f = []
    for j in range(1, n+1):
        lb = np.floor(j - (w_k) / 2)
        ub = np.ceil(j + (w_k) / 2)
        a = [Z[l] for l in range(int(lb), int(ub)) if l >= 0 and l <= len(Z) - 1]
        Z_f.append(np.max(a))
    return(np.asarray(Z_f))


def zhang(array, number_of_windows, dh_max, dh_0, c):
    """Implements Zhang et. al (2003)
    IN PROGRESS
    """
    w_k_list = list(map(window_size, range(number_of_windows)))
    w_k_min = w_k_list[0]
    A = array
    m = A.shape[0]
    n = A.shape[1]
    flag = np.zeros((m, n))
    for w_k in enumerate(w_k_list):
        if w_k[1] == w_k_min:
            w_k_1 = 0
        else:
            w_k_1 = w_k_list[w_k[0] - 1]
        for i in range(0, m):
            P_i = A[i,:]
            Z = P_i
            Z_e = erosion(Z, w_k[1], n)
            Z_f = dilation(Z_e, w_k[1], n)
            dh_t = dht(Z, w_k[1], w_k_1, dh_0, dh_max, c)
            for j in range(0, n):
                if Z[j] - Z_f[j] > dh_t:
                    flag[i, j] = w_k[1]
            P_i = Z_f
            A[i,:] = P_i
    return(flag)

def holder_func(array, grid):
    """
    This is just holding some things in development -bf 04/08/18
    :return:
    """
    flag = zhang(array, 3, 3, 1, 0.5)
    empty_y, empty_x = grid.empty_cells[:,0].astype(int), grid.empty_cells[:,1].astype(int)
    array[empty_x - 1, empty_y -1] = np.nan
    B = np.where(flag == 0, A, np.nan)
    plt.matshow(B)
