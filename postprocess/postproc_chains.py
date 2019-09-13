import logging
import numpy as np
import os
import sys
import glob
import yaml
import postprocess.postprocess_func as pp
from pyabc.utils import define_eps, gaussian_kde_scipy
import rans_ode.sumstat as sumstat
import postprocess.regression as regression


def main(args):

    # Initialization
    if len(args) > 1:
        input_path = args[1]
    else:
        input_path = os.path.join('../rans_ode', 'params.yml')

    input = yaml.load(open(input_path, 'r'))

    ### Paths
    path = input['path']
    path['calibration'] = os.path.join(path['output'], 'calibration/')
    N_chains = input['parallel_threads']
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING CHAINS ############')
    C_limits = np.loadtxt(os.path.join(path['calibration'], 'C_limits'))
    N_params = len(C_limits)
    files = np.empty(0)
    for chain in range(N_chains):
        files_onechain = glob.glob1(path['output'], "chain{}_*.npz".format(chain))
        files = np.hstack((files, np.array([os.path.join(path['output'], i) for i in files_onechain])))
    accepted = np.empty((0, N_params))
    dist = np.empty((0, 1))
    sum_stat = np.empty((0, len(np.load(files[0])['sumstat'][0])))
    logging.info('Loading data')
    for file in files:
        logging.debug('loading {}'.format(file))
        accepted = np.vstack((accepted, np.load(file)['C']))
        sum_stat = np.vstack((sum_stat, np.load(file)['sumstat']))
        dist = np.vstack((dist, np.load(file)['dist'].reshape((-1, 1))))
    # data = np.hstack((accepted, dist)).tolist()
    print(accepted.shape, sum_stat.shape, dist.shape)
    logging.info('\n')
    logging.info('\n############# MCMC-ABC ({} chains) ############'.format(N_chains))
    folder = os.path.join(path['output'], 'chains')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    num_bin_kde = 20
    num_bin_raw = 20
    ##############################################################################
    logging.info('2D raw marginals with {} bins per dimension'.format(num_bin_raw))
    H, C_final_joint = pp.calc_raw_joint_pdf(accepted, num_bin_raw, C_limits)
    np.savetxt(os.path.join(folder, 'C_final_joint{}'.format(num_bin_raw)), C_final_joint)
    pp.calc_marginal_pdf_raw(accepted, num_bin_raw, C_limits, folder)
    ##############################################################################
    logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde))
    Z, C_final_smooth = gaussian_kde_scipy(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
    np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde)), C_final_smooth)
    logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
    np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
    pp.calc_marginal_pdf_smooth(Z, num_bin_kde, C_limits, folder)

    for q in [0.05, 0.1, 0.25]:
        pp.marginal_confidence(N_params, folder, q)
        pp.marginal_confidence_joint(accepted, folder, q)
    ####################################################################################################################
    #
    ####################################################################################################################
    logging.info('\n############# Regression ############')
    path['regression_dist'] = os.path.join(path['output'], 'regression_dist')
    if not os.path.isdir(path['regression_dist']):
        os.makedirs(path['regression_dist'])
    path['regression_full'] = os.path.join(path['output'], 'regression_full')
    if not os.path.isdir(path['regression_full']):
        os.makedirs(path['regression_full'])
    Truth = sumstat.TruthData(valid_folder=path['valid_data'], case=input['case'])
    ind = np.argsort(accepted[:, -1])
    accepted = accepted[ind]
    sum_stat = sum_stat[ind]
    dist = dist[ind]
    x_list = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 1]
    for x in x_list:
        logging.info('\n')
        n = int(x * len(accepted))
        print('{} samples are taken for regression ({}% of {})'.format(n, x * 100, len(accepted)))
        samples = accepted[:n, :N_params]
        dist_reg = dist[:n, -1].reshape((-1, 1))
        ##########################################################################
        logging.info('Regression with distance')
        folder = os.path.join(path['regression_dist'], 'x_{}'.format(int(x*100)))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # delta = np.max(dist)
        new_samples = regression.regression_dist(samples, dist_reg, x=1)
        # new_samples = regression.regression(accepted, sum_stat, dist, Truth.sumstat_true, x=x)
        limits = np.empty((N_params, 2))
        for i in range(N_params):
            limits[i, 0] = np.min(new_samples[:, i])
            limits[i, 1] = np.max(new_samples[:, i])
            if limits[i, 1] - limits[i, 0] < 1e-8:
                logging.warning('too small new range')
                limits[i, 0] -= 0.001
                limits[i, 1] += 0.001
        print('new limits = ', limits)
        np.savetxt(os.path.join(folder, 'reg_limits'), limits)
        num_bin_kde_reg = 20
        Z, C_final_smooth = pp.gaussian_kde_scipy(new_samples, limits[:, 0], limits[:, 1], num_bin_kde_reg)
        print(C_final_smooth)
        np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde_reg)), C_final_smooth)
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
        pp.calc_marginal_pdf_smooth(Z, num_bin_kde_reg, limits, folder)
        for q in [0.05, 0.1, 0.25]:
            pp.marginal_confidence(N_params, folder, q)
            pp.marginal_confidence_joint(new_samples, folder, q)
        ##########################################################################
        logging.info('Regression with full summary statistics')
        folder = os.path.join(path['regression_full'], 'x_{}'.format(int(x*100)))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        new_samples = regression.regression_full(samples, sum_stat[:n], dist_reg, Truth.sumstat_true, x=1)
        # new_samples = regression.regression(accepted, sum_stat, dist, Truth.sumstat_true, x=x)
        limits = np.empty((N_params, 2))
        for i in range(N_params):
            limits[i, 0] = np.min(new_samples[:, i])
            limits[i, 1] = np.max(new_samples[:, i])
            if limits[i, 1] - limits[i, 0] < 1e-8:
                logging.warning('too small new range')
                limits[i, 0] -= 0.001
                limits[i, 1] += 0.001
        print('new limits = ', limits)
        np.savetxt(os.path.join(folder, 'reg_limits'), limits)
        num_bin_kde_reg = 20
        Z, C_final_smooth = pp.gaussian_kde_scipy(new_samples, limits[:, 0], limits[:, 1], num_bin_kde_reg)
        print(C_final_smooth)
        np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde_reg)), C_final_smooth)
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
        pp.calc_marginal_pdf_smooth(Z, num_bin_kde_reg, limits, folder)
        for q in [0.05, 0.1, 0.25]:
            pp.marginal_confidence(N_params, folder, q)
            pp.marginal_confidence_joint(new_samples, folder, q)
    logging.info('\n#############Done############')


if __name__ == '__main__':
    main(sys.argv)