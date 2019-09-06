import os
import numpy as np
from pyabc.utils import define_eps


def epanechnikov_kernel(dist, delta):

    ind_less = np.where(dist <= delta)
    out = np.zeros_like(dist)
    out[ind_less] = 3/4*(1-(dist[ind_less]/delta)**2)
    # print('min-max: ', np.min(out[ind_less][:-1]), np.max(out))
    return out.reshape((-1, ))


def local_linear_regression_dist(samples, dist, delta):

    N_samples, N_params = samples.shape
    X = np.hstack((np.ones((N_samples, 1)), dist))
    W = np.diag(epanechnikov_kernel(dist, delta))
    solution = np.linalg.inv(X.T @ W @ X)@X.T@W@samples
    print('solution = ', solution)
    alpha = solution[0]
    beta = solution[1:].reshape((-1, N_params))
    new_samples = samples - dist @ beta
    return new_samples


def local_linear_regression(samples, sumstat_dif, dist, delta):
    """
    :param samples: array of parameters of shape (N, N_params)
    :param sumstat_dif: array (S' - S_true) of shape (N, N_sum_stat)
    :param dist: array of distanses of shape (N, 1)
    :param delta: distance tolerance
    :return: array of new parameters of shape (N, N_params)
    """

    N_samples = samples.shape[0]
    X = np.hstack((np.ones((N_samples, 1)), sumstat_dif))
    W = np.diag(epanechnikov_kernel(dist, delta))
    solution = np.linalg.inv(X.T @ W @ X)@X.T@W@samples
    # alpha = solution[0]
    beta = solution[1:]
    new_samples = samples - sumstat_dif @ beta
    return new_samples


def regression_dist(samples, dist, x):

    data = np.hstack((samples, dist)).tolist()
    delta = define_eps(data, x)
    del data
    new_samples = local_linear_regression_dist(samples, dist, delta)
    return new_samples


def regression_full(samples, sum_stat, dist, sumstat_true, x):

    data = np.hstack((samples, dist)).tolist()
    delta = define_eps(data, x)
    del data
    new_samples = local_linear_regression(samples, np.abs(sum_stat - sumstat_true), dist,  delta)
    return new_samples