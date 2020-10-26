import numpy as np
import logging
import os
import sys
import itertools

basefolder = '../overflow_results/output4/'
N_params = 5    # Number of parameters in the model
ind_param_nominal = 10
# N_per_dim = 6     # Number of points is the same in each dimension
N_per_dim = [6, 7, 6, 7, 8]  # Number of points in each dimension
N_jobs = 60     # Number of processors (make N_jobs files with parameters)

C_nominal = [0.09, 0.5, 0.075, 0.0828, 0.31]   # beta_star, sigma_w1, beta_1, beta_2, a1
# np.savetxt(os.path.join(basefolder, 'c_array_nominal'), [C_nominal, C_nominal])

# C_limits = [[0.07, 0.18],   # beta_st
#             [0.2, 1.6],         # sigma_w1
#             [0.01, 0.23],       # beta1
#             [0.05, 1.6],       # beta2
#             [0.27, 0.36]]       # a1
b_bstar = True
C_limits = [[0.07, 0.18],   # beta_st
            [0.1, 2],         # sigma_w1
            [0.2, 3],       # beta1/beta*
            [-1, 13],       # beta2/beta*
            [0.24, 0.4]]       # a1

# C_limits = [[0.07, 0.2],   # beta_st
#             [0.14, 1.5],       # beta1/beta*
#             [-0.85, 20],       # beta2/beta*
#             [0.24, 0.36]]       # a1

# if need to add points in the end of file
add = True
N_per_dim2 = [6, 9, 6, 6, 9]
C_limits2 = [[0.07, 0.18],   # beta_st
            [0.1, 2.95],         # sigma_w1
            [0.2, 3],       # beta1/beta*
            [-1, 13],       # beta2/beta*
            [0.16, 0.4]]       # a1


def sampling_uniform_grid(N_each, C_limits):
    """ Create list of lists of N parameters manually (make grid) uniformly distributed on given interval
    :param N_each: scalar if the same number of points in each dimension
                   or list if number of points is different in each dimension
    :param C_limits: list of bounds in each dimension
    :return: list of N lists of sampled parameters
    """
    N_params = len(C_limits)
    if np.isscalar(N_each):     # if N_each is scalar then make a list of same numbers
        N_each = N_each*np.ones(N_params)
    # create list of 1D uniform grids for each direction
    grid_x = [uniform_grid(C_limits[i], N_each[i]) for i in range(N_params)]
    # create N-dimensional mesh of samples
    grid_mesh = np.meshgrid(*grid_x, indexing='ij')
    # convert into list of sampled parameters (lists)
    grid_ravel = np.empty((N_params, np.prod(N_each, dtype=np.int32)))
    for i in range(N_params):
        grid_ravel[i] = grid_mesh[i].ravel()
    grid_ravel = grid_ravel.T
    return grid_ravel


def uniform_grid(C_limit, N_each):
    """ Create 1D uniform grid. Divide in N_each interval and put points in the middle of interval
    :param C_limit: bounds
    :param N_each: number of points
    :return: numpy 1D array of uniformly spaced points
    """
    C_tmp = np.linspace(C_limit[0], C_limit[1], N_each + 1)
    C_tmp = C_tmp[:-1] + (C_tmp[1] - C_tmp[0]) / 2
    return C_tmp


def calc_N(N_total, N_jobs):
    """ Calculate number of parameters for each processor. Divide equally and then add remaining
    :param N_total: scalar Number of paraneters
    :param N_jobs: scalar Number of processors
    :return: numpy 1D array with number of parameters for each processor
    """
    N = np.array([int(N_total / N_jobs)] * N_jobs)
    if N_total % N_jobs != 0:
        reminder = N_total % N_jobs
        N[:reminder] += 1
    return N


def add_nominal_param(c_array, ind, nominal_values):
    c_array = np.array(c_array)
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
        # N = calc_N(N_total, N_jobs)
        if N_params == 4:
            C_array_add = add_nominal_param(C_array_add, ind_param_nominal, C_nominal)

        N_newsamples = len(C_array_add)
        print('N new samples = ', N_newsamples)
        N = calc_N(N_newsamples, N_jobs)
        C_array = np.array(C_array_add.copy())
        # C_array = np.vstack((C_array, C_array_add))
        N_samples = len(C_array)
        N = calc_N(N_samples, N_jobs)
        print("C_array length", len(C_array))
    ###################################################################################################################
    #
    ###################################################################################################################
    # print('Unique values per dimension')
    # for i in range(5):
    #     print(i, np.unique(C_array[:, i]))
    # if b_bstar:
    #     C_array[:, 2] *= C_array[:, 0]  # beta1
    #     C_array[:, 3] *= C_array[:, 0]  # beta2
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

    print('Unique values per dimension')
    for i in range(5):
        print(i, np.unique(C_array[:, i]))
    np.savetxt(os.path.join(basefolder, 'C_limits_init'), C_limits)
    # TODO: plot histogram to check that uniform

    print(f'{N_samples} samples in {N_jobs} jobs')
    print(f"{N_samples*7/60/N_jobs} hours for 1 job")


if __name__ == '__main__':
    main()