import logging
import numpy as np
import os
import sys
import glob
import yaml
import postprocess.postprocess_func as pp
from pyabc.kde import find_MAP_kde, kdepy_fftkde
from pyabc.distance import calc_err_norm2
from overflow.sumstat import TruthData, calc_sum_stat, GridData
from overflow.overflow_driver import Overflow
# import matplotlib as mpl
# mpl.use('pdf')
import matplotlib.pyplot as plt

def main():

    basefolder = '../'
    ### Paths
    path = {'output': os.path.join(basefolder, 'output/'), 'valid_data': '../overflow/valid_data/'}
    print('Path:', path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess0005')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    x_list = [0.3, 0.1, 0.05, 0.03, 0.01, 0.005]
    C_limits = np.array([[0.07, 0.11],   # beta_st
                         [0.3, 0.7],         # sigma_w1
                         [0.055, 0.09],       # beta1
                         [0.05, 0.135]])    # beta2
    np.savetxt(os.path.join(path['output'], 'C_limits_init'), C_limits)
    N_params = len(C_limits)
    folders = [os.path.join(path['output'], 'calibration_job{}'.format(i), ) for i in range(15)]

    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv'])
    sumstat_true = Truth.sumstat_true
    logging.info('Loading data')
    result = np.empty((0, len(sumstat_true)+5+1))
    N_total = 0
    for i, folder in enumerate(folders):
        N_total += len(np.loadtxt(os.path.join(folder, 'c_array_{}'.format(i))))
        with open(os.path.join(folder, 'result.dat')) as f:
            lines = f.readlines()
            for line in lines:
                d = np.fromstring(line[1:-1], dtype=float, sep=',')
                result = np.vstack((result, d))
        if N_total != len(result):
            print('Job {} did not finish ({} out of {})'.format(i, len(result), N_total))
    print(N_total, len(result))
    # all statistics
    dist = result[:, -1]
    # cp statistics
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     # dist[i] = calc_err_norm2(line[:Truth.length[0]], Truth.cp[:, 1])
    #     # dist[i] = calc_err_norm2(line[Truth.length[0]:Truth.length[1]], Truth.u_flat[:, 1])
    #     dist[i] = calc_err_norm2(line[Truth.length[1]:Truth.length[2]], Truth.uv_flat[:, 1])
    ind = np.argsort(dist)
    ####################################################################################################################
    #
    # ##################################################################################################################
    logging.info('\n############# Classic ABC ############')
    accepted = result[ind, :5]
    dist = dist[ind]
    for x in x_list:
        n = int(x * N_total)
        folder = os.path.join(path['output'], 'x_{}'.format(x * 100))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        logging.info('\n')
        print(folder)
        print('min dist = ', np.min(dist))
        accepted = accepted[:n, :N_params]
        dist = dist[:n]
        eps = np.max(dist)
        np.savetxt(os.path.join(folder, 'eps'), [eps])
        logging.info('x = {}, eps = {}, N accepted = {} (total {})'.format(x, eps, n, N_total))
        num_bin_kde = 20
        num_bin_raw = 20
        ##############################################################################
        logging.info('2D raw marginals with {} bins per dimension'.format(num_bin_raw))
        H, C_final_joint = pp.calc_raw_joint_pdf(accepted, num_bin_raw, C_limits)
        np.savetxt(os.path.join(folder, 'C_final_joint{}'.format(num_bin_raw)), C_final_joint)
        pp.calc_marginal_pdf_raw(accepted, num_bin_raw, C_limits, folder)
        del H
        # ##############################################################################
        logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde))
        Z = kdepy_fftkde(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
        C_final_smooth = find_MAP_kde(Z, C_limits[:, 0], C_limits[:, 1])
        np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde)), C_final_smooth)
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
        Z = np.load(os.path.join(folder, 'Z.npz'))['Z']
        pp.calc_marginal_pdf_smooth(Z, num_bin_kde, C_limits, folder)
        pp.calc_conditional_pdf_smooth(Z, folder)
        del Z
        for q in [0.05, 0.1, 0.25]:
            pp.marginal_confidence(N_params, folder, q)
            pp.marginal_confidence_joint(accepted, folder, q)
    del accepted, dist


if __name__ == '__main__':
    main()