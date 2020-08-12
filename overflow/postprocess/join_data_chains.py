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


def join_profiles(folders, cp, u, uv, u_surf):
    for i, folder in enumerate(folders):
        print('job {}'.format(i))
        cp = np.vstack((cp, np.fromfile(os.path.join(folder, 'cp_all.bin')).reshape(-1, 721)))
        u  = np.vstack((u, np.fromfile(os.path.join(folder, 'u_slice.bin')).reshape(-1, 800)))
        uv = np.vstack((uv, np.fromfile(os.path.join(folder, 'uv_slice.bin')).reshape(-1, 800)))
        u_surf = np.vstack((u_surf, np.fromfile(os.path.join(folder, 'u_surface.bin')).reshape(-1, 721)))
    return cp, u, uv, u_surf


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
    # basefolder = '../'
    # path = {'output': os.path.join(basefolder, 'overflow_results/chains_limits/'),
    #         'valid_data': '../overflow/valid_data/'}
    # N_jobs = [200]
    # raw_folders = ['chains_limits']
    ##################################
    basefolder = '../'
    path = {'output': os.path.join(basefolder, 'overflow_results/chains_limits/'),
            'valid_data': '../overflow/valid_data/'}
    N_jobs = 200
    data_folder = 'chains_limits'
    folders = [os.path.join(data_folder, 'chain_{}'.format(n), ) for n in range(N_jobs)]
    ##################################
    if not os.path.isdir(path['output']):
        os.makedirs(path['output'])
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler(os.path.join(path['output'], 'ABC_postprocess.log')), logging.StreamHandler()],
        level=logging.DEBUG)
    print('data folders to join:', data_folder)
    # We need truth statistics only to know length when loading data
    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv', 'x_separation'])
    sumstat_length = len(Truth.sumstat_true)
    ####################################################################################################
    logging.info('Loading c_array data')
    c_array = np.empty((0, N_PARAMS))
    sumstat_all = np.empty((0, sumstat_length))
    dist = []
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
    ####################################################################################################
    logging.info('Loading profiles data')

    cp_nominal = np.fromfile(os.path.join(path['nominal_data'], 'cp_all.bin'), dtype=float)
    u_nominal = np.fromfile(os.path.join(path['nominal_data'], 'u_slice.bin'), dtype=float)
    uv_nominal = np.fromfile(os.path.join(path['nominal_data'], 'uv_slice.bin'), dtype=float)
    u_surf_nominal = np.fromfile(os.path.join(path['nominal_data'], 'u_surface.bin'), dtype=float)

    logging.info('Loading data')
    cp_profile = np.empty((0, len(cp_nominal)))
    u_profile = np.empty((0, len(u_nominal)))
    uv_profile = np.empty((0, len(uv_nominal)))
    u_surf_profile = np.empty((0, len(u_surf_nominal)))

    cp_profile, u_profile, uv_profile, u_surf_profile = join_profiles(folders, cp_profile,
                                                                      u_profile, uv_profile, u_surf_profile)
    N_total = len(c_array)
    print(f'There are {N_total} samples in {N_PARAMS}D space')
    np.savez(os.path.join(path['output'], 'joined_profiles.npz'),
             cp=cp_profile, u=u_profile, uv=uv_profile, u_surface=u_surf_profile)
    ####################################################################################################


if __name__ == '__main__':
    main()