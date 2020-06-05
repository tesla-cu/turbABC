import logging
import numpy as np
import os
import shutil
import postprocess.postprocess_func as pp
from pyabc.kde import find_MAP_kde, kdepy_fftkde, gaussian_kde_scipy
from overflow.sumstat import TruthData
from plotting.plotting import plot_dist_pdf
from pyabc.distance import calc_err_norm2
from postprocess.postprocess_classic_abc import output_by_percent
N_jobs = 45


def load_data(folders, len_sumstat):
    N_total = 0
    result = np.empty((0, len_sumstat+5+1))   # + 5 parameters in the beginning and distance in the end
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
                                                                           N_total - len(result)))
    print(N_total, len(result))
    # N_total = len(result)
    print(N_total, 6**5)
    return result, N_total


def dist_by_sumstat(sumstat, sumstat_true):
    dist = np.empty(len(sumstat))
    for i, line in enumerate(sumstat):
        dist[i] = calc_err_norm2(line, sumstat_true)
    return dist



def main():

    basefolder = '../'
    ### Paths
    path = {'output': os.path.join(basefolder, 'overflow_results/output/'),
            'valid_data': '../overflow/valid_data/'}
    print('Path:', path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler(os.path.join(path['output'], 'ABC_postprocess.log')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    x_list = [0.3, 0.1, 0.05, 0.03, 0.01, 0.005]
    num_bin_kde = 20
    num_bin_raw = 10
    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv', 'x_separation'])
    sumstat_true = Truth.sumstat_true
    b_bstar = True
    C_limits = [[0.07, 0.18],  # beta_st
                [0.2, 1.6],  # sigma_w1
                [0.14, 1.27],  # beta1/beta*
                [0.27, 23],  # beta2/beta*
                [0.27, 0.36]]  # a1
    C_limits = np.array(C_limits)
    np.savetxt(os.path.join(path['output'], 'C_limits_init'), C_limits)
    N_params = len(C_limits)
    logging.info('Loading data')
    folders = [os.path.join(path['output'], 'calibration_job{}'.format(i), ) for i in range(N_jobs)]
    result, N_total = load_data(folders, len(sumstat_true))
    if b_bstar:
        result[:, 2] /= result[:, 0]    # beta1/beta*
        result[:, 3] /= result[:, 0]    # beta2/beta*
    ### tmp
    # unique_a = np.unique(result[:, 4])
    # print(unique_a)
    # print(np.diff(unique_a))
    # print(unique_a[0] - 1.5*np.diff(unique_a)[0], unique_a[0] - 2.5*np.diff(unique_a)[0])
    # exit()
    ###

    # ### taking slice
    # ind = np.where(result[:, 1] == 0.55)[0]
    # result = result[ind]
    # result = np.delete(result, 1, axis=1)
    # C_limits = np.delete(C_limits, 1, axis=0)
    # N_total = len(result)
    # N_params = len(C_limits)
    # # for i in range(len(Truth.length)):
    # #     Truth.length[i] -= 1
    # print('statistics length:', Truth.length)

    ####################################################################
    c_array = result[:, :N_params]
    sumstat_all = result[:, N_params:-1]
    # TODO: check if need to divide by norm
    print('/n########')
    # cp statistics
    print('cp statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[:Truth.length[0]], Truth.cp[:, 1])
    dist = dist_by_sumstat(sumstat_all[:, :Truth.length[0]], Truth.cp[:, 1])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_cp'), i_stat=0)
    # u statistics
    print('U statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[Truth.length[0]:Truth.length[1]], Truth.u_flat[:, 0])
    dist = dist_by_sumstat(sumstat_all[:, Truth.length[0]:Truth.length[1]], Truth.u_flat[:, 0])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_u'), i_stat=1)
    # uv statistics
    print('uv statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[Truth.length[1]:Truth.length[2]], -Truth.uv_flat[:, 0])
    dist = dist_by_sumstat(sumstat_all[:, Truth.length[1]:Truth.length[2]], -Truth.uv_flat[:, 0])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_uv'), i_stat=2)

    # cp + u statistics
    print('cp + U statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[:Truth.length[1]]/Truth.norm[:Truth.length[1]],
    #                              Truth.sumstat_true[:Truth.length[1]])
    dist = dist_by_sumstat(sumstat_all[:, :Truth.length[1]] / Truth.norm[:Truth.length[1]],
                           Truth.sumstat_true[:Truth.length[1]])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_cp_u'), i_stat=3)


    # x1 statistics
    print('x1 statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[Truth.length[2]], Truth.sumstat_true[Truth.length[2]])
    dist = dist_by_sumstat(sumstat_all[:, Truth.length[2]], Truth.sumstat_true[Truth.length[2]])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_x1'), i_stat=15)
    # x2 statistics
    print('x2 statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[Truth.length[2]+1], Truth.sumstat_true[Truth.length[2]+1])
    dist = dist_by_sumstat(sumstat_all[:, Truth.length[2]+1], Truth.sumstat_true[Truth.length[2]+1])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_x2'), i_stat=16)
    # x1 + x2 statistics
    print('x1+x2 statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, 5:-1]):
    #     dist[i] = calc_err_norm2(line[Truth.length[2]:], Truth.sumstat_true[Truth.length[2]:])
    dist = dist_by_sumstat(sumstat_all[:, Truth.length[2]:], Truth.sumstat_true[Truth.length[2]:])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_x1_x2'), i_stat=17)
    dist_x = dist.copy()

    # cp + u + uv statistics
    print('cp + U + uv statistics')
    # dist = np.empty(N_total)
    # for i, line in enumerate(result[:, N_params:-1]):
    #     print(len(line), Truth.length)
    #     dist[i] = calc_err_norm2(line[:Truth.length[2]] / Truth.norm[:Truth.length[2]],
    #                              Truth.sumstat_true[:Truth.length[2]])
    dist = dist_by_sumstat(sumstat_all[:, :Truth.length[2]] / Truth.norm[:Truth.length[2]],
                           Truth.sumstat_true[:Truth.length[2]])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_cp_u_uv'), i_stat=4)
    dist_all = dist.copy()

    # all statistics if dist(x1+x2)<0.5, else 0
    limits = [0.5, 0.25, 0.1, 0.05]
    for i, lim in enumerate(limits):
        print(f'all if less {lim} statistics')
        ind_nonzero = np.where(dist_x < lim)[0]
        result2 = result[ind_nonzero]
        dist = dist_all[ind_nonzero]
        print('nonzero: ', len(ind_nonzero))
        output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                          os.path.join(path['output'], f'postprocess_all_if_less_{lim}'), i_stat=5 + i)



    # # dist = np.zeros(N_total)
    # # ind_nonzero = []
    # # for i, line in enumerate(result[:, 5:-1]):
    # #     if calc_err_norm2(line[Truth.length[2]:], Truth.sumstat_true[Truth.length[2]:]) < 0.5:
    # #         dist[i] = calc_err_norm2(line[:Truth.length[2]] / Truth.norm[:Truth.length[2]],
    # #                                  Truth.sumstat_true[:Truth.length[2]])
    # #         ind_nonzero.append(i)
    # ind_nonzero = np.where(dist_x < 0.5)[0]
    # # ind_nonzero = np.array(ind_nonzero)
    # result2 = result[ind_nonzero]
    # dist = dist_all[ind_nonzero]
    #
    # print('nonzero: ', len(ind_nonzero))
    # output_by_percent(result2[:, N_params], dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_all_if_less_05'))






    # # if dist(x1+x2)<0.5, else 0
    # ind_nonzero = []
    # for i, line in enumerate(result[:, 5:-1]):
    #     if calc_err_norm2(line[Truth.length[2]:], Truth.sumstat_true[Truth.length[2]:]) < 0.5:
    #         ind_nonzero.append(i)
    # ind_nonzero = np.array(ind_nonzero)
    # accepted = result[ind_nonzero, :N_params]
    # print('nonzero: ', len(ind_nonzero))
    # marginal(accepted, C_limits, 20, 10, os.path.join(path['output'], 'postprocess_if_less_05', 'x_100.0', ))

    # # (0.5 < x1 < 1) and (1 < x2 < 2), else 0
    # ind_nonzero = []
    # for i, line in enumerate(result[:, 5:-1]):
    #     x1 = line[Truth.length[2]]
    #     x2 = Truth.sumstat_true[Truth.length[2]+1]
    #     if (0.5 < x1 < 1) and (1 < x2 < 2):
    #         ind_nonzero.append(i)
    # ind_nonzero = np.array(ind_nonzero)
    # accepted = result[ind_nonzero, :N_params]
    # print('nonzero: ', len(ind_nonzero))
    # marginal(accepted, C_limits, 20, 10, os.path.join(path['output'], 'postprocess_if_good', 'x_100.0', ))


if __name__ == '__main__':
    main()
