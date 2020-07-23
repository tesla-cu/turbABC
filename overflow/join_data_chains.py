import logging
import numpy as np
import os
import shutil
from overflow.sumstat import TruthData
from pyabc.distance import calc_err_norm2

N_PARAMS = 4


def load_data(folders, len_sumstat):
    N_total, diff = 0, 0
    result = np.empty((0, N_PARAMS + len_sumstat + 1))   # + 5 parameters in the beginning and distance in the end
    for i, folder in enumerate(folders):
        print('job {}'.format(i))
        with open(os.path.join(folder, 'result.dat')) as f:
            lines = f.readlines()
            for line in lines:
                d = np.fromstring(line[1:-1], dtype=float, sep=',')
                result = np.vstack((result, d))
    return result


def define_limits_for_uniform(c_array):
    N_params = c_array.shape[1]
    C_limits = np.empty((N_params, 2))
    for i in range(N_params):
        unique = np.unique(c_array[:, i])
        dif = np.max(np.diff(unique))
        C_limits[i] = [unique[j] - (j+0.5)*dif for j in [0, -1]]
    return C_limits


# def join_limits(limits, N_params):
#     all_lower_limits = np.ones((N_params, 1)) * (1000)
#     all_upper_limits = np.ones((N_params, 1)) * (-1000)
#     for limit in limits:
#         for i in range(N_params):
#             all_lower_limits[i] = min(limit[i, 0], all_lower_limits[i])
#             all_upper_limits[i] = max(limit[i, 1], all_upper_limits[i])
#     return np.hstack((all_lower_limits, all_upper_limits))


def main():

    ######################################
    # Define parameters
    #####################################
    basefolder = '../'
    path = {'output': os.path.join(basefolder, 'overflow_results/chains_limits/'),
            'valid_data': '../overflow/valid_data/'}
    N_jobs = [200]
    raw_folders = ['chains_limits']
    ##################################
    if not os.path.isdir(path['output']):
        os.makedirs(path['output'])
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler(os.path.join(path['output'], 'ABC_postprocess.log')), logging.StreamHandler()],
        level=logging.DEBUG)
    data_folders = [os.path.join(basefolder, 'overflow_results', folder) for folder in raw_folders]
    print('data folders to join:', data_folders)
    # We need truth statistics only to know length when loading data
    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv', 'x_separation'])
    sumstat_length = len(Truth.sumstat_true)

    logging.info('Loading data')
    c_array = np.empty((0, N_PARAMS))
    sumstat_all = np.empty((0, sumstat_length))
    dist = []
    for i, data_folder in enumerate(data_folders):
        folders = [os.path.join(data_folder, 'chain_{}'.format(n), ) for n in range(N_jobs[i])]
        result = load_data(folders, sumstat_length)  # to check length need c_array_* files


        c_array = np.vstack((c_array, result[:, :N_PARAMS]))
        sumstat_all = np.vstack((sumstat_all, result[:, N_PARAMS:-1]))
        dist.append(np.min(result[:, -1]))
    print('min stores dist:', np.min(dist))

    N_total = len(c_array)
    print(f'There are {N_total} samples in {N_PARAMS}D space')
    C_limits = define_limits_for_uniform(c_array)
    np.savez(os.path.join(path['output'], 'joined_data.npz'),
             c_array=c_array, sumstat_all=sumstat_all, C_limits=C_limits, N_total=N_total)

    ####################################################################################################


if __name__ == '__main__':
    main()