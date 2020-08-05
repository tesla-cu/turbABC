import os
import numpy as np


# from sklearn.metrics import r2_score


def calc_r2_score(y, fit_line):
    """ The same as sklearn.metrics.r2_score(y, fit_line)
    :param y: y coordinates of samples
    :param fit_line: y coordinates of fitted line
    :return: R^2 score
    """
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y-fit_line)**2)
    R = 1 - ss_res/ss_tot
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
    """ Separately finds linear fit for each summary statistic,
        calculate R^2 scores of fits and returns indices of summary statistic with R^2 score > 0.85.
    :param samples:
    :param sumstat_dif:
    :param dist:
    :param delta:
    :param folder:
    :return:
    """
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
    print('good_indices', good_sumstat_indices)
    print(len(good_sumstat_indices), sumstat_dif.shape[1])
    print(np.min(sumstat_dif, axis=0) < 0, np.max(sumstat_dif, axis=0) > 0)
    print(np.logical_and(np.min(sumstat_dif, axis=0) < 0, np.max(sumstat_dif, axis=0) > 0))

    return good_sumstat_indices


def local_linear_regression(samples, sumstat_dif, dist, delta):
    """
    :param samples: array of parameters of shape (N, N_params)
    :param sumstat_dif: array (S' - S_true) of shape (N, N_sum_stat)
    :param dist: array of distances of shape (N, 1)
    :param delta: distance tolerance
    :return: array of new parameters of shape (N, N_params)
    """
    N_samples = samples.shape[0]
    X = np.hstack((np.ones((N_samples, 1)), sumstat_dif))
    print('X.shape', X.shape)
    # W_1d = epanechnikov_kernel(dist, delta)
    W_1d = np.ones(len(dist))   # uniform kernel
    X_T_W = X.T*W_1d
    solution = np.linalg.inv(X_T_W @ X)@X_T_W@samples
    print('solution = ', solution)
    alpha = solution[0]
    beta = solution[1:]
    print('alpha', alpha)
    print('beta', beta)
    # just test
    solution = np.linalg.pinv(X)@samples
    print(np.linalg.pinv(X).shape)
    print(samples.shape)
    # print('solution = ', solution)
    # alpha = solution[0]
    # beta = solution[1:]
    # print('alpha', alpha)
    # print('beta', beta)
    print(beta.shape, sumstat_dif.shape, np.outer(sumstat_dif,  beta).shape, samples.shape)
    new_samples = samples - np.outer(sumstat_dif,  beta)
    return new_samples, solution


def regression(samples, sum_stat_diff, dist, x, folder):

    # data = np.hstack((samples, dist)).tolist()
    # delta = define_eps(data, x)
    # del data
    if sum_stat_diff.shape[1] != 1:
        indices = test_statistics(samples, sum_stat_diff,  dist, 1, folder)
        # plot_compare_truth.plot_experiment('../rans_plots/', indices)
        sum_stat_diff = sum_stat_diff[:, indices]
        print('sum_stat_diff.shape', sum_stat_diff.shape)
    new_samples = local_linear_regression(samples, sum_stat_diff, dist, 1)
    return new_samples