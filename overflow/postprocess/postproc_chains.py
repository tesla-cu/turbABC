import logging
import numpy as np
import os
from overflow.sumstat import TruthData
from pyabc.distance import calc_err_norm2, calc_err_norm1
from postprocess.postprocess_classic_abc import marginal


def dist_by_sumstat(sumstat, sumstat_true):
    dist = np.empty(len(sumstat))
    for i, line in enumerate(sumstat):
        dist[i] = calc_err_norm2(line, sumstat_true)
    return dist


def main():

    basefolder = '../../'
    ### Paths
    path = {'output': os.path.join(basefolder, 'overflow_results/chains_limits_final/'),
            'valid_data': '../../overflow/valid_data/'}
    postproc_folder = os.path.join(path['output'], 'postprocess')
    if not os.path.isdir(postproc_folder):
        os.makedirs(postproc_folder)
    print('Path:', path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler(os.path.join(path['output'], 'ABC_postprocess.log')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    data_file = 'joined_data.npz'
    num_bin_kde = 50
    mirror = False

    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv', 'x_separation'])
    sumstat_true = Truth.sumstat_true
    norm = Truth.norm
    print('statistics length:', Truth.length)

    logging.info('Loading data from .npz')
    c_array = np.load(os.path.join(path['output'], data_file))['c_array']
    ### tmp
    # sumstat_all = np.load(os.path.join(path['output'], data_file))['sumstat_all']
    C_limits = np.load(os.path.join(path['output'], data_file))['C_limits']
    np.savetxt(os.path.join(path['output'], 'C_limits_init'), C_limits)
    N_total, N_params = c_array.shape
    logging.info(f'There are {N_total} samples total in {N_params}D space')
    # !!! stored summary statistics are not divided by norm
    ####################################################################
    marginal(c_array, C_limits, num_bin_kde, 0, postproc_folder, mirror)


if __name__ == '__main__':
    main()
