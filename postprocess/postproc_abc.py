import logging
import numpy as np
import os
import sys
import glob
import yaml
import postprocess.postprocess_func as pp
from pyabc.utils import define_eps
from pyabc.kde import gaussian_kde_scipy
import rans_ode.sumstat as sumstat
import postprocess.regression as regression


def main(args):

    # Initialization
    if len(args) > 1:
        input_path = args[1]
    else:
        input_path = os.path.join('../runs_abc/', 'params.yml')

    input = yaml.load(open(input_path, 'r'))

    ### Paths
    # path = input['path']
    # path = {'output': os.path.join('../runs_abc/', 'output/'), 'valid_data': '../rans_ode/valid_data/'}
    path = {'output': os.path.join('../', 'output/'), 'valid_data': '../rans_ode/valid_data/'}

    print(path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess0005')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    x_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3]
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    N_params = len(C_limits)
    if len(C_limits.shape) < 2:
        N_params = 1
    files_abc = glob.glob1(path['output'], "classic_abc*.npz")
    files = [os.path.join(path['output'], i) for i in files_abc]
    accepted = np.empty((0, N_params))
    dist = np.empty((0, 1))
    sum_stat = np.empty((0, len(np.load(files[0])['sumstat'][0])))
    logging.info('Loading data')

    for file in files:
        logging.debug('loading {}'.format(file))
        accepted = np.vstack((accepted, np.load(file)['C'][:, :N_params]))
        sum_stat = np.vstack((sum_stat, np.load(file)['sumstat']))
        dist = np.vstack((dist, np.load(file)['dist'].reshape((-1, 1))))
    data = np.hstack((accepted, dist)).tolist()
    logging.info('\n############# Classic ABC ############')
    for x in x_list:
        logging.info('\n')
        folder = os.path.join(path['output'], 'x_{}'.format(x * 100))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        print(folder)
        print('min dist = ', np.min(dist))
        eps = define_eps(data, x)
        np.savetxt(os.path.join(folder, 'eps'), [eps])
        abc_accepted = accepted[np.where(dist < eps)[0]]
        logging.info('x = {}, eps = {}, N accepted = {} (total {})'.format(x, eps, len(abc_accepted), len(dist)))
        num_bin_kde = 20
        num_bin_raw = 20
        ##############################################################################
        logging.info('2D raw marginals with {} bins per dimension'.format(num_bin_raw))
        H, C_final_joint = pp.calc_raw_joint_pdf(abc_accepted, num_bin_raw, C_limits)
        np.savetxt(os.path.join(folder, 'C_final_joint{}'.format(num_bin_raw)), C_final_joint)
        pp.calc_marginal_pdf_raw(abc_accepted, num_bin_raw, C_limits, folder)
        # ##############################################################################
        logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde))
        Z, C_final_smooth = gaussian_kde_scipy(abc_accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
        np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde)), C_final_smooth)
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
        pp.calc_marginal_pdf_smooth(Z, num_bin_kde, C_limits, folder)

        for q in [0.05, 0.1, 0.25]:
            pp.marginal_confidence(N_params, folder, q)
            pp.marginal_confidence_joint(abc_accepted, folder, q)
    ####################################################################################################################
    #
    # ####################################################################################################################
    logging.info('\n############# Regression ############')
    path['regression_dist'] = os.path.join(path['output'], 'regression_dist')
    path['regression_full'] = os.path.join(path['output'], 'regression_full')
    if not os.path.isdir(path['regression_dist']):
        os.makedirs(path['regression_dist'])
    if not os.path.isdir(path['regression_full']):
        os.makedirs(path['regression_full'])
    Truth = sumstat.TruthData(valid_folder=path['valid_data'], case=input['case'])
    ind = np.argsort(dist)
    accepted = accepted[ind]
    sum_stat = sum_stat[ind]
    dist = dist[ind]
    for x in x_list:
        logging.info('\n')
        n = int(x * len(accepted))
        print('{} samples are taken for regression ({}% of {})'.format(n, x * 100, len(accepted)))
        samples = accepted[:n, :N_params]
        dist_reg = dist[:n, -1].reshape((-1, 1))
        ##########################################################################
        logging.info('Regression with distance')
        folder = os.path.join(path['regression_dist'], 'x_{}'.format(x*100))
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
    logging.info('\n#############Done############')


if __name__ == '__main__':
    main(sys.argv)