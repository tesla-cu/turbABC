import numpy as np
import logging
import os
import sys
import itertools

basefolder = '../output/'
N_params = 4
N_per_dim = 10
N_jobs = 15

C_nominal = [0.09, 0.5, 0.09/1.2, 0.09/1.0870, 0.31]   # beta_star, sigma_w1, beta_1, beta_2, a1

C_limits = [[0.07, 0.11],   # beta_st
            [0.3, 0.7],         # sigma_w1
            [0.055, 0.09],       # beta1
            [0.05, 0.135]]       # beta2
            # [0.31, 0.40],       # a1

# if need to add points in the end of file
add = 1
N_per_dim2 = 14
C_limits2 = [[0.054, 0.11],   # beta_st
             [0.3, 0.86],         # sigma_w1
             [0.041, 0.09],       # beta1
             [0.016, 0.135]]       # beta2


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


def calc_N(N_total, N_jobs):
    N = np.array([int(N_total / N_jobs)] * N_jobs)
    if N_total % N_jobs != 0:
        reminder = N_total % N_jobs
        N[:reminder] += 1
    return N


def main():

    N_total = N_per_dim**N_params
    N = calc_N(N_total, N_jobs)
    C_array = sampling_uniform_grid(N_per_dim, C_limits)
    if N_params < 5:
        c_nominal_array = np.array([C_nominal[N_params:]]*len(C_array))
        C_array = np.hstack((C_array, c_nominal_array))
    print('N samples = ', len(C_array))
    ###################################################################################################################
    #
    ###################################################################################################################
    if add:
        C_array2 = sampling_uniform_grid(N_per_dim2, C_limits2)
        C_array_add = []
        for c in C_array2:
            bool_inside = True in np.logical_and(np.array(C_limits)[:, 0] < c, c < np.array(C_limits)[:, 1])
            if not bool_inside:
                C_array_add.append(c)
        N_total += len(C_array_add)
        N = calc_N(N_total, N_jobs)
        if N_params < 5:
            c_nominal_array = np.array([C_nominal[N_params:]]*len(C_array_add))
            C_array_add = np.hstack((C_array_add, c_nominal_array))
        print('N samples = ', len(C_array_add))
        C_array = np.vstack((C_array, C_array_add))
        print("C_array.shape:", C_array.shape)
    ###################################################################################################################
    #
    ###################################################################################################################
    for i in range(N_jobs):
        start, end = np.sum(N[:i]), np.sum(N[:i+1])
        print('job {}: from {} to {}'.format(i, start, end))
        dir = os.path.join(basefolder, 'calibration_job{}'.format(i))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.savetxt(os.path.join(dir, 'c_array_{}'.format(i)), C_array[start:end])


if __name__ == '__main__':
    main()