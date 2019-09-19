import numpy as np
import logging
from time import time
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE
from pyabc.utils import timer


def grid_for_kde(a, b, num_bin_joint):

    dim = len(a)
    grid_x = [np.linspace(a[i], b[i], num_bin_joint + 1) for i in range(dim)]
    grid_mesh = np.meshgrid(*grid_x, indexing='ij')
    grid_ravel = np.empty((dim, (num_bin_joint+1)**dim))
    for i in range(dim):
        grid_ravel[i] = grid_mesh[i].ravel()
    return grid_mesh, grid_ravel


def bw_from_kdescipy(data, method='scott'):

    kde = gaussian_kde(data.T, bw_method=method)
    f = kde.covariance_factor()
    data_std = np.std(data, axis=0)
    bw = f * data_std
    print('Scott: f, bw = ', f, bw)
    return bw


def find_MAP_kde(Z, grid_mesh):
    C_map = []
    ind = np.argwhere(Z == np.max(Z))
    for i in ind:
        C_map.append([grid_mesh[j][i[j]] for j in range(len(grid_mesh))])
    return C_map


def gaussian_kde_scipy(data, a, b, num_bin_joint):

    logging.info('Scipy: Gaussian KDE {} dimensions with {} bins per dimension'.format(len(a), num_bin_joint))

    kde = gaussian_kde(data.T, bw_method='scott')
    # # evaluate on a regular grid
    grid_mesh, grid_ravel = grid_for_kde(a, b, num_bin_joint)
    time1 = time()
    Z = kde.evaluate(grid_ravel)
    Z = Z.reshape(grid_mesh[0].shape)
    time2 = time()
    timer(time1, time2, "Time for gaussian_kde_scipy")
    C_map = find_MAP_kde(Z, grid_mesh)

    return Z, C_map


# TODO: finish this function and test
def kdepy_fftkde(data, a, b, num_bin_joint):
    logging.info('KDEpy.FFTKDe: Gaussian KDE {} dimensions'.format(len(a)))
    time1 = time()
    bandwidth = bw_from_kdescipy(data, 'scott')[0]
    kde = FFTKDE(kernel='gaussian', bw=bandwidth)
    kde.fit(data)
    grid_mesh, Z = kde.evaluate()
    time2 = time()
    return grid_mesh, Z,