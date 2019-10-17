import logging
import numpy as np
import os
import sys
import glob
import yaml
import postprocess.postprocess_func as pp
from pyabc.utils import define_eps
from pyabc.kde import gaussian_kde_scipy, kdepy_fftkde
import rans_ode.sumstat as sumstat
from postprocess.regression import regression


def load_c(files, N_params):
    accepted = np.empty((0, N_params))
    for file in files:
        logging.debug('loading C from {}'.format(file))
        accepted = np.vstack((accepted, np.load(file)['C'][:, :N_params]))
    return accepted


def load_sum_stat(files):
    sum_stat = np.empty((0, len(np.load(files[0])['sumstat'][0])))
    for file in files:
        logging.debug('loading sum_stat from {}'.format(file))
        sum_stat = np.vstack((sum_stat, np.load(file)['sumstat']))
    return sum_stat


def load_dist(files):
    dist = np.empty((0, 1))
    for file in files:
        logging.debug('loading dist from {}'.format(file))
        dist = np.vstack((dist, np.load(file)['dist'].reshape((-1, 1))))
    return dist


def new_limits(new_samples, N_params):
    limits = np.empty((N_params, 2))
    for i in range(N_params):
        limits[i, 0] = np.min(new_samples[:, i])
        limits[i, 1] = np.max(new_samples[:, i])
        if limits[i, 1] - limits[i, 0] < 1e-8:
            logging.warning('too small new range')
            limits[i, 0] -= 0.001
            limits[i, 1] += 0.001
    print('new limits = ', limits)
    return limits


def main(args):

    # Initialization
    if len(args) > 1:
        input_path = args[1]
    else:
        input_path = os.path.join('../runs_abc/', 'params.yml')

    input = yaml.load(open(input_path, 'r'))

    ### Paths
    # path = input['path']
    path = {'output': os.path.join('../runs_abc/', 'output/'), 'valid_data': '../rans_ode/valid_data/'}
    # path = {'output': os.path.join('../', 'output/'), 'valid_data': '../rans_ode/valid_data/'}
    print(path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess0005')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    x_list = [0.3, 0.1, 0.05, 0.03, 0.01, 0.005]
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    N_params = len(C_limits)
    files_abc = glob.glob1(path['output'], "classic_abc*.npz")
    files = [os.path.join(path['output'], i) for i in files_abc]

    logging.info('Loading data')
    accepted = load_c(files, N_params)
    dist = load_dist(files)
    ind = np.argsort(dist[:, 0])
    N_total = len(accepted)
    ####################################################################################################################
    #
    # # ##################################################################################################################
    logging.info('\n############# Classic ABC ############')
    accepted = accepted[ind]
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
        # Z, C_final_smooth = gaussian_kde_scipy(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
        Z, C_final_smooth = kdepy_fftkde(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
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
    exit()
    # ####################################################################################################################
    # #
    # # ##################################################################################################################
    # logging.info('\n############# Regression ############')
    # # regression_type = ['regression_dist', 'regression_full']
    # regression_type = ['regression_full']
    # num_bin_kde_reg = 20
    # samples = load_c(files, N_params)[ind]
    # dist = load_dist(files)[ind]
    # for type in regression_type:
    #     path[type] = os.path.join(path['output'], type)
    #     if not os.path.isdir(path[type]):
    #         os.makedirs(path[type])
    #     if type == 'regression_full':
    #             sum_stat = load_sum_stat(files)[ind]
    #     for x in x_list:
    #         n = int(x * N_total)
    #         folder = os.path.join(path[type], 'x_{}'.format(x*100))
    #         if not os.path.isdir(folder):
    #             os.makedirs(folder)
    #         logging.info('\n')
    #         logging.info('Regression {}'.format(type))
    #         logging.info('{} samples are taken for regression ({}% of {})'.format(n, x * 100, N_total))
    #         samples = samples[:n, :N_params]
    #         dist = dist[:n]
    #         if type == 'regression_full':
    #             sum_stat = sum_stat[:n]
    #         ##########################################################################
    #         if type == 'regression_dist':
    #             new_samples, solution = regression(samples, dist, dist, x=1)
    #         else:
    #             Truth = sumstat.TruthData(valid_folder=path['valid_data'], case=input['case'])
    #             new_samples, solution = regression(samples, sum_stat - Truth.sumstat_true, dist, x=1)
    #         limits = new_limits(new_samples, N_params)
    #         np.savetxt(os.path.join(folder, 'reg_limits'), limits)
    #         Z, C_final_smooth = kdepy_fftkde(new_samples, limits[:, 0], limits[:, 1], num_bin_kde_reg)
    #         # Z, C_final_smooth = pp.gaussian_kde_scipy(new_samples, limits[:, 0], limits[:, 1], num_bin_kde_reg)
    #         logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
    #         np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde_reg)), C_final_smooth)
    #         np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
    #         pp.calc_marginal_pdf_smooth(Z, num_bin_kde_reg, limits, folder)
    #         del Z
    #         for q in [0.05, 0.1, 0.25]:
    #             pp.marginal_confidence(N_params, folder, q)
    #             pp.marginal_confidence_joint(new_samples, folder, q)
    # logging.info('\n#############Done############')


if __name__ == '__main__':
    main(sys.argv)