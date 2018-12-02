import numpy as np
import logging
import itertools

def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>1}:{:0>2}:{:05.2f} \t {}".format(int(hours), int(minutes), seconds, label))


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


def axisymmetric_expansion():
    S = np.zeros(6)
    s = -(1 / 2.45)
    S[0] = s        # S11
    S[1] = -s / 2   # S22
    S[2] = -s / 2   # S33
    return S


def axisymmetric_contraction():
    S = np.zeros(6)
    s = (1 / 0.179)
    S[0] = s
    S[1] = -s / 2
    S[2] = -s / 2
    return S


def pure_shear():
    S = np.zeros(6)
    S[3] = (1/0.296)/2  # S12
    return S


def plane_strain():
    S = np.zeros(6)
    S[0] = 1/2
    S[1] = -1/2
    return S


# def periodic(t):
#     S = np.zeros(6)
#     s0 = 3.3
#     beta = 0.125
#     S[3] = (s0 / 2) * np.sin(beta * s0 * t)  #applied shear
#     return S


def covariance_recursive(x, t, cov_prev, mean_prev, s_d):
    mean_new = t / (t + 1) * mean_prev + 1 / (t + 1) * x
    cov = (t - 1) / t * cov_prev + \
          s_d / t * (t * np.outer(mean_prev, mean_prev) - (t + 1) * np.outer(mean_new, mean_new) + np.outer(x, x))
    return cov, mean_new