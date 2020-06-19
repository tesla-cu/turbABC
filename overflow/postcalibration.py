import logging
import numpy as np
import os
import postprocess.postprocess_func as pp
import pyabc.glob_var as g
from overflow.posstproc_overflow import load_data, dist_by_sumsta
from pyabc.distance import calc_err_norm2
from overflow.sumstat import TruthData, calc_sum_stat, GridData
from pyabc.abc_alg import calibration_postprocess1 as calibration_postprocess1

N_jobs = 45


def main():

    basefolder = '../'
    ### Paths
    path = {'output': os.path.join(basefolder, 'output/'), 'valid_data': '../overflow/valid_data/',
            'calibration': os.path.join(basefolder, 'output', 'calibration', )}
    print('Path:', path)
    g.path = path
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess0005')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# Loading data ############')
    C_limits = np.array([[0.07, 0.11],   # beta_st
                         [0.3, 0.7],         # sigma_w1
                         [0.055, 0.09],       # beta1
                         [0.05, 0.135]])    # beta2
    np.savetxt(os.path.join(path['output'], 'C_limits_init'), C_limits)
    N_params = len(C_limits)
    folders = [os.path.join(path['output'], 'calibration_job{}'.format(i), ) for i in range(N_jobs)]

    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv'])
    sumstat_true = Truth.sumstat_true

    logging.info('Loading data')
    result = np.empty((0, len(sumstat_true)+5+1))  # 5 parameters in the beginning and distance in the end
    N_total = 0
    for i, folder in enumerate(folders):
        print('job {}'.format(i))
        N_total += len(np.loadtxt(os.path.join(folder, 'c_array_{}'.format(i))))
        with open(os.path.join(folder, 'result.dat')) as f:
            lines = f.readlines()
            for line in lines:
                d = np.fromstring(line[1:-1], dtype=float, sep=',')
                result = np.vstack((result, d))
        if N_total != len(result):
            print('Job {} did not finish ({} out of {}), diff = {}'.format(i, len(result), N_total,
                                                                           N_total-len(result)))
    print(N_total, len(result))
    # all statistics
    # dist = result[:, -1]
    # cp statistics
    dist = np.empty(N_total)
    for i, line in enumerate(result[:, 5:-1]):
        # all 3 statistics equally treated
        dist[i] = calc_err_norm2(line, Truth.sumstat_true)
        ## only Cp statistics
        # dist[i] = calc_err_norm2(line[:Truth.length[0]], Truth.cp[:, 1])
        ## only velocity statistics
        # dist[i] = calc_err_norm2(line[Truth.length[0]:Truth.length[1]], Truth.u_flat[:, 0])
        ## only uv statistics
        # dist[i] = calc_err_norm2(line[Truth.length[1]:Truth.length[2]], -Truth.uv_flat[:, 0])
        # if i<10:
        #     plot_results(-Truth.uv_flat[:, 0], line[Truth.length[1]:Truth.length[2]], path['output'], str(i))
    result[:, -1] = dist
    # parameters for defining a new range
    x = 0.01
    phi = 1
    new_limits = calibration_postprocess1(result, x, phi, C_limits)
    print(C_limits)

if __name__ == '__main__':
    main()