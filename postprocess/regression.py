import os
import numpy as np
from pyabc.utils import define_eps
from sklearn.metrics import r2_score
# from plotting import plot_compare_truth


def calc_r2_score(y, fit_line):
    """
    The same as sklearn.metrics.r2_score(y, fit_line)
    :param y:
    :param fit_line:
    :return:
    """
    y_mean = np.mean(y)
    SS_tot = np.sum((y - y_mean)**2)
    # SS_reg = np.sum((fit_line - y_mean)**2)
    SS_res = np.sum((y-fit_line)**2)
    R = 1 - SS_res/SS_tot
    return R


def line(alpha, beta, x):
    return alpha + x @ beta


def epanechnikov_kernel(dist, delta):

    ind_less = np.where(dist <= delta)
    out = np.zeros_like(dist)
    out[ind_less] = 3/4*(1-(dist[ind_less]/delta)**2)
    # print('min-max: ', np.min(out[ind_less][:-1]), np.max(out))
    return out.reshape((-1, ))


def test_statistics(samples, sumstat_dif, dist, delta, folder):

    N_sumstat = sumstat_dif.shape[1]
    print('N_stat =', N_sumstat)
    indices = np.arange(N_sumstat)
    # indices = np.where(np.logical_and(np.min(sumstat_dif, axis=0) < 0, np.max(sumstat_dif, axis=0) > 0))[0]
    R2_scores = np.empty(len(indices))
    for i, ind in enumerate(indices):
        s = sumstat_dif[:, ind].reshape((-1, 1))
        _, solution = local_linear_regression(samples, s, dist, delta)
        np.savetxt(os.path.join(folder, 'solution{}'.format(ind)), solution)
        alpha = solution[0]
        beta = solution[1:]
        line_fit = line(alpha, beta, s)
        R2_scores[i] = calc_r2_score(samples, line_fit)
    good_sumstat_indices = indices[np.where(R2_scores > 0.85)[0]]
    print(good_sumstat_indices)
    print(len(good_sumstat_indices), sumstat_dif.shape[1])
    print(np.min(sumstat_dif, axis=0) < 0, np.max(sumstat_dif, axis=0) > 0)
    print(np.logical_and(np.min(sumstat_dif, axis=0) < 0, np.max(sumstat_dif, axis=0) > 0))


    return good_sumstat_indices


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
    # print('solution = ', solution)
    # alpha = solution[0]
    beta = solution[1:]
    new_samples = samples - sumstat_dif @ beta
    return new_samples, solution


def regression(samples, sum_stat_diff, dist, x, folder):

    # data = np.hstack((samples, dist)).tolist()
    # delta = define_eps(data, x)
    # del data
    indices = test_statistics(samples, sum_stat_diff,  dist, 1, folder)
    plot_compare_truth.plot_experiment('../plots/', indices)
    sum_stat_diff = sum_stat_diff[:, indices]
    print(sum_stat_diff.shape)
    new_samples = local_linear_regression(samples, sum_stat_diff, dist, 1)
    return new_samples