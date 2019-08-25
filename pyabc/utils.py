import numpy as np
import logging
import itertools
from scipy.stats import gaussian_kde
from time import time
import pyabc.glob_var as g


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>1}:{:0>2}:{:05.2f} \t {}".format(int(hours), int(minutes), seconds, label))


def take_safe_log10(x):
    """Takes natural logarithm and put g.TINY number where x = 0"""
    log_fill = np.empty_like(x)
    log_fill.fill(g.TINY_log)
    log = np.log10(x, out=log_fill, where=x > g.TINY)
    return log


def check_output_size(N, N_params, sumstat_size):
    writing_size = 2 ** 30/8 - 1e5   # max array size for np.savez is 2^30 bytes
    biggest_array_size = N * max(N_params, sumstat_size)
    n, r = np.divmod(N, writing_size)
    return int(n), int(r), int(writing_size)


def uniform_grid(C_limits, N_each):
    C_tmp = np.linspace(C_limits[0], C_limits[1], N_each + 1)
    C_tmp = C_tmp[:-1] + (C_tmp[1] - C_tmp[0]) / 2
    return C_tmp


def sampling_random(N_total, C_limits):
    """ Random sampling with uniform distribution.
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    C_array = np.random.random(size=(N_total, N_params))
    for i in range(N_params):
        C_array[:, i] = C_array[:, i] * (C_limits[i, 1] - C_limits[i, 0]) + C_limits[i, 0]
    C_array = C_array.tolist()
    return C_array


def sampling_uniform_grid(N_each, C_limits):
    """ Create list of lists of N parameters manually (make grid) uniformly distributed on given interval
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    C = np.empty((N_params, N_each))
    for i in range(N_params):
        C[i, :] = uniform_grid(C_limits[i], N_each)
    permutation = itertools.product(*C)
    C_array = list(map(list, permutation))
    logging.debug('Form C_array as uniform grid: {} samples\n'.format(len(C_array)))
    return C_array


def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf


def covariance_recursive_haario(x, t, cov_prev, mean_prev):
    mean_new = mean_prev + 1 / (t + 1) * (x-mean_prev)
    cov = (t - 1) / t * cov_prev + \
          1 / t * (t * np.outer(mean_prev, mean_prev) - (t + 1) * np.outer(mean_new, mean_new) + np.outer(x, x))
    return cov, mean_new


def covariance_recursive(x, t, cov_prev, mean_prev):
    delta = (x-mean_prev)
    mean_new = mean_prev + 1 / (t + 1) * delta
    cov = (t - 1) / t * cov_prev + 1/(t + 1) * np.outer(delta, delta)
    return cov, mean_new


def gaussian_kde_scipy(data, a, b, num_bin_joint):
    dim = len(a)
    C_map = []
    print(dim, data.shape, a, b)
    data_std = np.std(data, axis=0)
    kde = gaussian_kde(data.T, bw_method='scott')
    f = kde.covariance_factor()
    bw = f * data_std
    print('Scott: f, bw = ', f, bw)
    # kde = gaussian_kde(data.T, bw_method='silverman')
    # f = kde.covariance_factor()
    # bw = f * data_std
    # print('Silverman: f, bw = ', f, bw)
    # kde.set_bandwidth(bw_method=kde.factor / 4.)
    # f = kde.covariance_factor()
    # bw = f * data_std
    # print('f, bw = ', f, bw)

    time1 = time()
    # # evaluate on a regular grid
    xgrid = np.linspace(a[0], b[0], num_bin_joint + 1)
    if dim == 1:
        Z = kde.evaluate(xgrid)
        Z = Z.reshape(xgrid.shape)
        ind = np.argwhere(Z == np.max(Z))
        for i in ind:
            C_map.append(xgrid[i])
    elif dim == 2:
        ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='ij')
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z = Z.reshape(Xgrid.shape)
    elif dim == 3:
        ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
        zgrid = np.linspace(a[2], b[2], num_bin_joint + 1)
        Xgrid, Ygrid, Zgrid = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel(), Zgrid.ravel()]))
        Z = Z.reshape(Xgrid.shape)
        ind = np.argwhere(Z == np.max(Z))
        for i in ind:
            C_map.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]]])
    elif dim == 4:
        ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
        zgrid = np.linspace(a[2], b[2], num_bin_joint + 1)
        z4grid = np.linspace(a[3], b[3], num_bin_joint + 1)
        Xgrid, Ygrid, Zgrid, Z4grid = np.meshgrid(xgrid, ygrid, zgrid, z4grid, indexing='ij')
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel(), Zgrid.ravel(), Z4grid.ravel()]))
        Z = Z.reshape(Xgrid.shape)
        ind = np.argwhere(Z == np.max(Z))
        for i in ind:
            C_map.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]], z4grid[i[3]]])
    else:
        print("gaussian_kde_scipy: Wrong number of dimensions (dim)")
    time2 = time()
    timer(time1, time2, "Time for gaussian_kde_scipy")
    return Z, C_map