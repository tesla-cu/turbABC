import os
import numpy as np
from pyabc.utils import define_eps


def epanechnikov_kernel(dist, delta):

    ind_less = np.where(dist <= delta)
    out = np.zeros_like(dist)
    out[ind_less] = 3/4*(1-(dist[ind_less]/delta)**2)
    # print('min-max: ', np.min(out[ind_less][:-1]), np.max(out))
    return out.reshape((-1, ))



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
    # W_1d = epanechnikov_kernel(dist, delta)
    W_1d = np.ones(len(dist))
    X_T_W = X.T*W_1d
    solution = np.linalg.inv(X_T_W @ X)@X_T_W@samples
    print('solution = ', solution)
    # alpha = solution[0]
    beta = solution[1:]
    new_samples = samples - sumstat_dif @ beta
    return new_samples, solution


def regression(samples, sum_stat_diff, dist, x):

    # data = np.hstack((samples, dist)).tolist()
    # delta = define_eps(data, x)
    # del data
    new_samples = local_linear_regression(samples, sum_stat_diff, dist, 1)
    return new_samples