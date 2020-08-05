import logging
import numpy as np
import os
import sys
import glob
import postprocess.postprocess_func as pp
from pyabc.utils import define_eps
from pyabc.kde import gaussian_kde_scipy, kdepy_fftkde, find_MAP_kde
from postprocess.postprocess_classic_abc import output_by_percent
from postprocess.regression import regression
import rans_ode.sumstat as sumstat


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
            logging.warning('new range is too small')
            limits[i, 0] -= 0.001
            limits[i, 1] += 0.001
    print('new limits = ', limits)
    return limits


def main():
    path = {'output': os.path.join('../', 'rans_output/'), 'valid_data': '../rans_ode/valid_data/'}
    print(path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess0005')),
                  logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    x_list = [0.3, 0.1, 0.05, 0.03, 0.01]
    # x_list = [0.3]
    num_bin_kde = 20
    num_bin_raw = 10
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    if len(np.array(C_limits).shape) < 2:
        N_params = 1
    else:
        N_params = len(C_limits)
    files_abc = glob.glob1(path['output'], "classic_abc*.npz")
    files = [os.path.join(path['output'], i) for i in files_abc]

    logging.info('Loading data')

    dist_unsorted = load_dist(files)
    ind = np.argsort(dist_unsorted[:, 0])
    samples = load_c(files, N_params)[ind]
    N_total, _ = samples.shape
    dist = dist_unsorted[ind]
    # ##################################################################################################################
    # #
    # # ################################################################################################################
    logging.info('\n############# Regression ############')
    # # regression_type = ['regression_dist', 'regression_full']
    # regression_type = ['regression_full']
    regression_type = ['regression_dist']
    for type in regression_type:
        path[type] = os.path.join(path['output'], type)
        if not os.path.isdir(path[type]):
            os.makedirs(path[type])
        # if type == 'regression_full':
        #     sum_stat = load_sum_stat(files)[ind]
        for x in x_list:
            n = int(x * N_total)
            folder = os.path.join(path[type], 'x_{}'.format(x*100))
            if not os.path.isdir(folder):
                os.makedirs(folder)
            logging.info('\n')
            logging.info('Regression {}'.format(type))
            logging.info('{} samples are taken for regression ({}% of {})'.format(n, x * 100, N_total))
            samples = samples[:n, :N_params]
            dist = dist[:n]
            # if type == 'regression_full':
            #     sum_stat = sum_stat[:n]
            ##########################################################################
            if type == 'regression_dist':
                new_samples, solution = regression(samples, dist, dist, x=1, folder=folder)
            # else:
            #     Truth = sumstat.TruthData(valid_folder=path['valid_data'], case=input['case'])
            #     new_samples, solution = regression(samples, sum_stat - Truth.sumstat_true, dist, x=1)
            limits = new_limits(new_samples, N_params)
            np.savetxt(os.path.join(folder, 'reg_limits'), limits)
            Z= kdepy_fftkde(new_samples, limits[:, 0], limits[:, 1], num_bin_kde)
            # Z, C_final_smooth = pp.gaussian_kde_scipy(new_samples, limits[:, 0], limits[:, 1], num_bin_kde_reg)
            C_final_smooth = find_MAP_kde(Z, C_limits[:, 0], C_limits[:, 1])
            logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
            np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde)), C_final_smooth)
            np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
            pp.calc_marginal_pdf_smooth(Z, num_bin_kde, limits, folder)
            pp.calc_conditional_pdf_smooth(Z, folder)
            del Z
            # for q in [0.05, 0.1, 0.25]:
            #     pp.marginal_confidence(N_params, folder, q)
            #     pp.marginal_confidence_joint(new_samples, folder, q)
    logging.info('\n#############Done############')


if __name__ == '__main__':
    main()