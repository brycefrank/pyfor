import numpy as np

# PARAMETERS
# TODO put this in function calls later
dh_0 = 1
dh_max = 3
c = 1

def window_size(k):
    k = 1
    b = 2
    return(2 * k * b + 1)


def dhmax(elev_array):
    return(np.max(elev_array) - np.min(elev_array))


def slope(elev_array, w_k, w_k_1):
    """
    Returns the slope coefficient s for a given elev_aray and w_k
    """
    return(dhmax(elev_array) / ((w_k - w_k_1) / 2))


def dht(elev_array, w_k, w_k_1):
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
    return(Z_f)

def dilation(Z, w_k, n):
    Z_f = []
    for j in range(1, n+1):
        lb = np.floor(j - (w_k) / 2)
        ub = np.ceil(j + (w_k) / 2)
        a = [Z[l] for l in range(int(lb), int(ub)) if l >= 0 and l <= len(Z) - 1]
        Z_f.append(np.max(a))
    return(Z_f)


def zhang(array, k):
    """Implements Zhang et. al (2003)
    IN PROGRESS
    """
    w_k_list = list(map(window_size, range(k)))
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
            Z_f = erosion(Z, w_k[1], n)
            Z_f = dilation(Z_f, w_k[1], n)
            P_i = Z_f
            print(Z_f)
            A[i,:] = P_i
            dh_t = dht(Z, w_k[1], w_k_1)
            for j in range(0, n):
                if Z[j] - Z_f[j] > dh_t:
                    flag[i, j] = w_k[1]
    return(flag)