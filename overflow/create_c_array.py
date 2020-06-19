import numpy as np
import logging
import os
import sys
import itertools

basefolder = '../overflow_results/output4/'
N_params = 4
ind_param_nominal = 1
N_per_dim = 10
# N_per_dim = [6, 7, 6, 7, 8]
N_jobs = 120

C_nominal = [0.09, 0.5, 0.075, 0.0828, 0.31]   # beta_star, sigma_w1, beta_1, beta_2, a1
# np.savetxt(os.path.join(basefolder, 'c_array_nominal'), [C_nominal, C_nominal])

# C_limits = [[0.07, 0.18],   # beta_st
#             [0.2, 1.6],         # sigma_w1
#             [0.01, 0.23],       # beta1
#             [0.05, 1.6],       # beta2
#             [0.27, 0.36]]       # a1
b_bstar = True
# C_limits = [[0.07, 0.18],   # beta_st
#             [0.08333333333333325, 1.48333333],         # sigma_w1
#             [0.14, 1.27],       # beta1/beta*
#             [-3.5183333333333326, 23],       # beta2/beta*
#             [0.24, 0.36]]       # a1

C_limits = [[0.07, 0.2],   # beta_st
            [0.14, 1.5],       # beta1/beta*
            [-0.85, 20],       # beta2/beta*
            [0.24, 0.36]]       # a1

# if need to add points in the end of file
add = 0
N_per_dim2 = [6, 7, 6, 7, 8]
C_limits2 = [[0.07, 0.18],   # beta_st
             [-0.03333333333333344, 1.6],         # sigma_w1
             [0.14, 1.27],       # beta1/beta*
             [-3.5183333333333326, 23],       # beta2/beta*
             [0.24, 0.36]]       # a1


def sampling_uniform_grid(N_each, C_limits):
    """ Create list of lists of N parameters manually (make grid) uniformly distributed on given interval
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    if np.isscalar(N_each):
        N_each = N_each*np.ones(N_params)
    grid_x = [uniform_grid(C_limits[i], N_each[i]) for i in range(N_params)]
    grid_mesh = np.meshgrid(*grid_x, indexing='ij')
    grid_ravel = np.empty((N_params, np.prod(N_each, dtype=np.int32)))
    for i in range(N_params):
        grid_ravel[i] = grid_mesh[i].ravel()
    grid_ravel = grid_ravel.T
    print(grid_ravel.shape)
    return grid_ravel


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


def add_nominal_param(c_array, ind, nominal_values):
    c_nominal_array = (np.ones((len(c_array), 1)) * nominal_values[ind])
    return np.hstack((c_array[:, :ind], c_nominal_array, c_array[:, ind:]))


def main():

    N_total = N_per_dim**N_params if np.isscalar(N_per_dim) else np.prod(N_per_dim)

    N = calc_N(N_total, N_jobs)
    C_array = sampling_uniform_grid(N_per_dim, C_limits)
    if N_params == 4:
        C_array = add_nominal_param(C_array, ind_param_nominal, C_nominal)

    N_samples = len(C_array)
    print('N samples = ', N_samples)
    ###################################################################################################################
    #
    ###################################################################################################################
    if add:
        C_array2 = sampling_uniform_grid(N_per_dim2, C_limits2)
        C_array_add = []
        for c in C_array2:
            bool_outside = False in np.logical_and(np.array(C_limits)[:, 0] < c, c < np.array(C_limits)[:, 1])
            if bool_outside:
                C_array_add.append(c)
        N_total += len(C_array_add)
        N = calc_N(N_total, N_jobs)
        if N_params == 4:
            C_array = add_nominal_param(C_array, ind_param_nominal, C_nominal)
        N_samples = len(C_array_add)
        print('N samples = ', N_samples)
        C_array = np.vstack((C_array, C_array_add))
        print("C_array.shape:", C_array.shape)
    ###################################################################################################################
    #
    ###################################################################################################################
    if b_bstar:
        C_array[:, 2] *= C_array[:, 0]  # beta1
        C_array[:, 3] *= C_array[:, 0]  # beta2
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

    np.savetxt(os.path.join(basefolder, 'C_limits_init'), C_limits)
    # TODO: plot histogram to check that uniform
    print(f'{N_samples} samples in {N_jobs} jobs')
    print(f"{N_samples*7/60/N_jobs} hours for 1 job")


if __name__ == '__main__':
    main()