import numpy as np
import logging
import itertools

import pyabc.glob_var as g


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>1}:{:0>2}:{:05.2f} \t {}".format(int(hours), int(minutes), seconds, label))


def take_safe_log10(x):
    """Takes logarithm with base 10 and put g.TINY number where x = 0"""
    log_fill = np.empty_like(x)
    log_fill.fill(g.TINY_log)
    log = np.log10(x, out=log_fill, where=x > g.TINY)
    return log


def take_safe_log(x):
    """Takes natural logarithm and put g.TINY number where x = 0"""
    log_fill = np.empty_like(x)
    log_fill.fill(g.TINY_log)
    log = np.log(x, out=log_fill, where=x > g.TINY)
    return log


def check_output_size(N, N_params, sumstat_size):
    writing_size = 2 ** 30/8 - 1e5   # max array size for np.savez is 2^30 bytes
    biggest_row_size = max(N_params, sumstat_size)
    n = N // int(writing_size/biggest_row_size)
    r = N % int(writing_size/biggest_row_size)
    return int(n), int(r), int(writing_size/biggest_row_size)


def define_eps(array, x):
    if isinstance(array, np.ndarray):
        ind = np.argsort(array[:, -1])
        array = array[ind]
    elif isinstance(array, list):
        array.sort(key=lambda y: y[-1])
        array = np.array(array)
    eps = np.percentile(array, q=x*100, axis=0)[-1]
    return eps


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
    pdf, edges = np.histogram(array, bins=bins, range=range, density=1)
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




