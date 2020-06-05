import numpy as np
import logging
from time import time
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE
from pyabc.utils import timer


def grid_for_kde(a, b, num_bin):
    """
    Create n-dimensional grid mesh and ravel mesh based on boundaries and number of bins per dimension.
    Note: number of points per dimension is (num_bin+1)
    :param a: list of left boundaries
    :param b: list of right boundaries
    :param num_bin: number of bins (cells) per dimension
    :return: list of np.meshgrid's (one for each dimension), list of ravel meshes (one for each dimension)
    """
    if not hasattr(a, "__len__"):
        a, b = [a], [b]
    dimension = len(a)
    grid_x = [np.linspace(a[i], b[i], num_bin + 1) for i in range(dimension)]
    grid_mesh = np.meshgrid(*grid_x, indexing='ij')
    grid_ravel = np.empty((dimension, (num_bin+1)**dimension))
    for i in range(dimension):
        grid_ravel[i] = grid_mesh[i].ravel()
    return grid_mesh, grid_ravel


def bw_from_kdescipy(data, method='scott'):
    """ Calculate bandwidth for Gaussian Kernel Density Estimation function, using scipy.stats.gaussian_kde.
        Use Scott's rule by default.
    :param data: array of parameter samples
    :param method: 'scott' or 'silverman'
    :return: array of bandwidth (different for each dimension of data)
    """
    kde = gaussian_kde(data.T, bw_method=method)
    f = kde.covariance_factor()
    data_std = np.std(data, axis=0)
    bw = f * data_std
    # print('Scott: f, bw = ', f, bw)
    return bw


def find_MAP_kde(Z, a, b):
    """ Find the parameter, which gives the maximum of a posteriori distibution (MAP).
        Can handles multiple maximum values.
    :param Z: a posteriori distribution function: numpy array of shape (num_bin+1, )*dimensions
    :param a: list of left boundaries for parameter space
    :param b: list of right boundaries for parameter space
    :return: list of MAP parameters: list of lists
    """
    if hasattr(a, "__len__"):
        n_params = len(a)
    else:
        n_params = 1
        a, b = [a], [b]
    c_map = []
    num_points = Z.shape[0]
    indices_max = np.argwhere(Z == np.max(Z))     # can be more then one max
    for ind in indices_max:
        c = np.empty_like(a, dtype=float)
        for n in range(n_params):
            grid = np.linspace(a[n], b[n], num_points, endpoint=True)
            c[n] = grid[ind[n]]
        c_map.append(c)
    return c_map


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

    return Z


def kdepy_fftkde(data, a, b, num_bin_joint):
    """ Calculate Kernel Density Estimation (KDE) using KDEpy.FFTKDE.
    Note: KDEpy.FFTKDE can do only symmetric kernel (accept only scalar bandwidth).
    We map data to [-1, 1] domain to make bandwidth independent of parameter range and more symmetric
    and use mean of list bandwidths (different bandwidth for each dimension)
    calculated usinf Scott's rule and scipy.stats.gaussian_kde
    :param data: array of parameter samples
    :param a: list of left boundaries
    :param b: list of right boundaries
    :param num_bin_joint: number of bins (cells) per dimension in estimated posterior
    :return: estimated posterior of shape (num_bin_joint, )*dimensions
    """

    N_params = len(data[0])
    logging.info('KDEpy.FFTKDe: Gaussian KDE {} dimensions'.format(N_params))
    time1 = time()
    a = np.array(a)-1e-10
    b = np.array(b)+1e-10
    data = 2 * (data - a) / (b - a) - 1     # transform data to be [-1, 1], since gaussian is the same in all directions
    bandwidth = bw_from_kdescipy(data, 'scott')
    _, grid_ravel = grid_for_kde(-1*np.ones(N_params), np.ones(N_params), num_bin_joint)
    kde = FFTKDE(kernel='gaussian', bw=np.mean(bandwidth))
    kde.fit(data)
    Z = kde.evaluate(grid_ravel.T)
    Z = Z.reshape((num_bin_joint + 1, )*N_params)
    time2 = time()
    timer(time1, time2, "Time for kdepy_fftkde")
    return Z
