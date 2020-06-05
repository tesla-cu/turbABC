import logging
import numpy as np
import os
import shutil
import postprocess.postprocess_func as pp
from pyabc.kde import find_MAP_kde, kdepy_fftkde, gaussian_kde_scipy
from plotting.plotting import plot_dist_pdf


def output_by_percent(result, dist, C_limits, x_list, num_bin_raw, num_bin_kde, output_folder, i_stat):
    logging.info('\n############# Classic ABC ############')
    N_params, N_total = len(C_limits), len(result)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # plot_dist_pdf(output_folder, dist, 0.1)

    ind = np.argsort(dist)
    accepted = result[ind]
    dist = dist[ind]
    for x in x_list:
        n = int(x * N_total)
        folder = os.path.join(output_folder, 'x_{}'.format(x * 100))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        logging.info('\n')
        print(folder)
        print('min dist = {} at {}'.format(np.min(dist), accepted[0]))
        accepted = accepted[:n, :N_params]
        dist = dist[:n]
        eps = np.max(dist)
        np.savetxt(os.path.join(folder, 'eps'), [eps])
        logging.info('x = {}, eps = {}, N accepted = {} (total {})'.format(x, eps, n, N_total))
        marginal(accepted, C_limits, num_bin_kde, num_bin_raw, folder)
    del accepted, dist
    collect_MAP_values(output_folder, x_list, num_bin_kde)
    path_up = os.path.dirname(os.path.normpath(output_folder))
    stat_name = os.path.basename(os.path.normpath(output_folder))[12:]
    shutil.copy(os.path.join(output_folder, 'MAP_values'), os.path.join(path_up, f'MAP_values_{stat_name}'))
    job_folder = os.path.join(path_up, 'MAP_jobs', f'calibration_job{i_stat}')
    if not os.path.isdir(job_folder):
        os.makedirs(job_folder)
        shutil.copy(os.path.join(output_folder, 'MAP_values'), os.path.join(job_folder, f'MAP_values_{stat_name}'))
        shutil.copy(os.path.join(output_folder, 'MAP_values'), os.path.join(job_folder, f'c_array_{i_stat}'))


def marginal(accepted, C_limits, num_bin_kde, num_bin_raw, folder):

    if not os.path.isdir(folder):
        os.makedirs(folder)
    ##############################################################################
    logging.info('2D raw marginals with {} bins per dimension'.format(num_bin_raw))
    H, C_final_joint = pp.calc_raw_joint_pdf(accepted, num_bin_raw, C_limits)
    np.savetxt(os.path.join(folder, 'C_final_joint{}'.format(num_bin_raw)), C_final_joint)
    pp.calc_marginal_pdf_raw(accepted, num_bin_raw, C_limits, folder)
    del H
    # ##############################################################################
    logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde))
    # mirrored_data, _ = pp.mirror_data_for_kde(accepted, C_limits[:, 0], C_limits[:, 1])
    # left, right = np.min(mirrored_data, axis=0), np.max(mirrored_data, axis=0)
    # Z = kdepy_fftkde(mirrored_data, left, right, num_bin_kde)
    Z = kdepy_fftkde(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
    C_final_smooth = find_MAP_kde(Z, C_limits[:, 0], C_limits[:, 1])
    np.savetxt(os.path.join(folder, 'C_final_smooth' + str(num_bin_kde)), C_final_smooth)
    # np.savetxt(os.path.join(folder, 'mirrored_limits'), [left, right])
    logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
    # ##############################################################################
    np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
    Z = np.load(os.path.join(folder, 'Z.npz'))['Z']
    pp.calc_marginal_pdf_smooth(Z, num_bin_kde, C_limits, folder)
    pp.calc_conditional_pdf_smooth(Z, folder)
    del Z
    # N_params = len(C_limits)
    # for q in [0.05, 0.1, 0.25]:
    #     pp.marginal_confidence(N_params, folder, q)
    #     pp.marginal_confidence_joint(accepted, folder, q)


def collect_MAP_values(output_folder, x_list, n_bin_smooth):
    MAP = []
    for x in x_list:
        folder = os.path.join(output_folder, 'x_{}'.format(x * 100))
        MAP.append(np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(n_bin_smooth))))

    if len(MAP[0]) == 4:
        MAP = np.hstack((np.array(MAP), np.ones(len(MAP)).reshape(-1, 1)*0.31))
        np.savetxt(os.path.join(output_folder, 'MAP_values'), MAP)
    return
