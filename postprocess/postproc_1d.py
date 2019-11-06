import logging
import numpy as np
import os
import sys
import glob
import yaml
import postprocess.postprocess_func as pp
from pyabc.utils import define_eps
from pyabc.kde import gaussian_kde_scipy, kdepy_fftkde, find_MAP_kde
import rans_ode.sumstat as sumstat
from postprocess.regression import regression
import pyabc.utils as utils


def main(args):

    # Initialization
    if len(args) > 1:
        input_path = args[1]
    else:
        input_path = os.path.join('../rans_ode', 'params.yml')

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
    np.savetxt(os.path.join(path['output'], '1d_dist_scatter'), data)
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
        print(sum_stat.shape, sum_stat[np.where(dist < eps)[0], :].shape)
        np.savez(os.path.join(path['output'], '1d_dist_scatter_{}'.format(x)), C=abc_accepted,
                 dist=dist[np.where(dist < eps)].reshape((-1, 1)), sumstat=sum_stat[np.where(dist < eps)[0], :])
        num_bin_kde = 100
        num_bin_raw = 20
        ##############################################################################
        logging.info('1D raw histogram with {} bins per dimension'.format(num_bin_raw))
        x, y = utils.pdf_from_array_with_x(abc_accepted, bins=num_bin_raw, range=C_limits)
        np.savetxt(os.path.join(folder, 'histogram'), [x, y])
        np.savetxt(os.path.join(folder, 'C_final_raw{}'.format(num_bin_raw)), [x[np.argmax(y)]])
        # ##############################################################################
        logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde))
        Z = kdepy_fftkde(abc_accepted, [C_limits[0]], [C_limits[1]], num_bin_kde)
        C_final_smooth = find_MAP_kde(Z, C_limits[0], C_limits[1])
        grid = np.linspace(C_limits[0]-1e-10, C_limits[1]+1e-10, num_bin_kde+1)
        # Z, C_final_smooth = gaussian_kde_scipy(abc_accepted, [C_limits[0]], [C_limits[1]], num_bin_kde)
        # grid = np.linspace(C_limits[0], C_limits[1], num_bin_kde+1)
        np.savetxt(os.path.join(folder, 'C_final_smooth'), C_final_smooth)
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z, grid=grid)

        # for q in [0.05, 0.1, 0.25]:
        #     pp.marginal_confidence(N_params, folder, q)
        #     pp.marginal_confidence_joint(abc_accepted, folder, q)
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
    ind = np.argsort(dist[:, 0])
    accepted = accepted[ind]
    sum_stat = sum_stat[ind]
    dist = dist[ind]
    for x in x_list:
        logging.info('\n')
        n = int(x * len(accepted))
        print('{} samples are taken for regression ({}% of {})'.format(n, x * 100, len(accepted)))
        samples = accepted[:n, :N_params]
        dist_reg = dist[:n, -1].reshape((-1, 1))
        #########################################################################
        logging.info('Regression with distance')
        folder = os.path.join(path['regression_dist'], 'x_{}'.format(x*100))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # new_samples, solution = regression(samples, (sum_stat[:n] - Truth.sumstat_true).reshape((-1, 1)),
        #                                    dist_reg, 1, folder)
        new_samples, solution = regression(samples, dist_reg, dist_reg, 1, folder)
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
        logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde_reg))
        Z = kdepy_fftkde(new_samples, [limits[:, 0]], [limits[:, 1]], num_bin_kde_reg)
        C_final_smooth = find_MAP_kde(Z, [limits[:, 0]], [limits[:, 1]])
        grid = np.linspace(limits[0, 0]-1e-10, limits[0, 1]+1e-10, num_bin_kde_reg+1)
        # Z, C_final_smooth = gaussian_kde_scipy(abc_accepted, [C_limits[0]], [C_limits[1]], num_bin_kde)
        # grid = np.linspace(C_limits[0], C_limits[1], num_bin_kde+1)
        np.savetxt(os.path.join(folder, 'C_final_smooth'), C_final_smooth[0])
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth[0]))
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z, grid=grid)
        # for q in [0.05, 0.1, 0.25]:
        #     pp.marginal_confidence(N_params, folder, q)
        ##########################################################################
        logging.info('Regression with full summary statistics')
        folder = os.path.join(path['regression_full'], 'x_{}'.format(x*100))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        new_samples, _ = regression(samples, sum_stat[:n] - Truth.sumstat_true, dist_reg, 1, folder)
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
        logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde_reg))
        Z = kdepy_fftkde(new_samples, [limits[:, 0]], [limits[:, 1]], num_bin_kde_reg)
        C_final_smooth = find_MAP_kde(Z, [limits[:, 0]], [limits[:, 1]])
        grid = np.linspace(limits[0, 0]-1e-10, limits[0, 1]+1e-10, num_bin_kde_reg+1)
        # Z, C_final_smooth = gaussian_kde_scipy(abc_accepted, [C_limits[0]], [C_limits[1]], num_bin_kde)
        # grid = np.linspace(C_limits[0], C_limits[1], num_bin_kde+1)
        np.savetxt(os.path.join(folder, 'C_final_smooth'), C_final_smooth[0])
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth[0]))
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z, grid=grid)
        # for q in [0.05, 0.1, 0.25]:
        #     pp.marginal_confidence(N_params, folder, q)
    logging.info('\n#############Done############')


if __name__ == '__main__':
    main(sys.argv)