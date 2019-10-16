import numpy as np
import logging
import os
import sys
import itertools

basefolder = '../output/'
N_params = 2
N_per_dim = 3
N_jobs = 3

C_nominal = [0.09, 0.5, 0.31, 0.0828, 0.075]   # beta_star, sigma_w1, a1, beta_2, beta_1

C_limits = [[0.0784, 0.1024],   # beta_st
            [0.3, 0.7]]         # sigma_w1
            # [0.31, 0.40],       # a1
            # [1.05, 1.45],       # beta_st/beta2
            # [1.19, 1.31]]       # beta_st/beta1


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


def uniform_grid(C_limit, N_each):
    C_tmp = np.linspace(C_limit[0], C_limit[1], N_each + 1)
    C_tmp = C_tmp[:-1] + (C_tmp[1] - C_tmp[0]) / 2
    return C_tmp


def main():

    N_total = N_per_dim**N_params
    N = np.array([int(N_total/N_jobs)]*N_jobs)
    if N_total%N_jobs != 0:
        reminder = N_total % N_jobs
        N[:reminder] += 1
    C_array = sampling_uniform_grid(N_per_dim, C_limits)
    if N_params < 5:
        c_nominal_array = np.array([C_nominal[N_params:]]*len(C_array))
        C_array = np.hstack((C_array, c_nominal_array))
    print('N samples = ', len(C_array))
    for i in range(N_jobs):
        start, end = np.sum(N[:i]), np.sum(N[:i+1])
        print('job {}: from {} to {}'.format(i, start, end))
        dir = os.path.join(basefolder, 'calibration_job{}'.format(i))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.savetxt(os.path.join(dir, 'c_array_{}'.format(i)), C_array[start:end])

if __name__ == '__main__':
    main()