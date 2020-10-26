import logging
import numpy as np
import os
from overflow.sumstat import TruthData
from pyabc.distance import calc_err_norm2, calc_err_norm1
from pyabc.abc_alg import calibration_postprocess1
from postprocess.postprocess_classic_abc import output_by_percent
from plotting.plotting import plot_dist_pdf

def dist_by_sumstat(sumstat, sumstat_true):
    dist = np.empty(len(sumstat))
    for i, line in enumerate(sumstat):
        dist[i] = calc_err_norm2(line, sumstat_true)
    return dist


def main():

    basefolder = '../../'
    ### Paths
    path = {'output': os.path.join(basefolder, 'overflow_results/output_inflow/'),
            'valid_data': '../../overflow/valid_data/'}
    print('Path:', path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler(os.path.join(path['output'], 'ABC_postprocess.log')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    data_file = 'joined_data.npz'
    x_list = [0.3, 0.2, 0.1, 0.05, 0.03]
    num_bin_kde = 15
    # num_bin_raw = (6, 6+7, 6, 7, 8)
    num_bin_raw = (6, 9, 6, 6, 9)
    # num_bin_raw = [12]*4
    mirror = False

    Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv', 'x_separation'])
    sumstat_true = Truth.sumstat_true
    norm = Truth.norm
    print('statistics length:', Truth.length)


    logging.info('Loading data from .npz')
    c_array = np.load(os.path.join(path['output'], data_file))['c_array']
    ### tmp
    sumstat_all = np.load(os.path.join(path['output'], data_file))['sumstat_all']
    C_limits = np.load(os.path.join(path['output'], data_file))['C_limits']
    np.savetxt(os.path.join(path['output'], 'C_limits_init'), C_limits)
    N_total, N_params = c_array.shape
    logging.info(f'There are {N_total} samples total in {N_params}D space')
    # !!! stored summary statistics are not divided by norm
    # ####################################################################
    # x1+x2 statistics
    logging.info('x1+x2 statistics')
    dist = dist_by_sumstat(sumstat_all[:, Truth.length[2]:], sumstat_true[Truth.length[2]:])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_x1_x2'), i_stat=17, mirror=mirror)
    dist_x = dist.copy()
    # ####################################################################
    # cp + u + uv statistics
    logging.info('cp + U + uv statistics')
    dist = dist_by_sumstat(sumstat_all[:, :Truth.length[2]] / norm[:Truth.length[2]], sumstat_true[:Truth.length[2]])
    output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_cp_u_uv'), i_stat=4, mirror=mirror)
    np.savez(os.path.join(path['output'], f'postprocess_cp_u_uv', 'dist_data.npz'), dist=dist)
    dist_all = dist.copy()
    ####################################################################
    # all statistics if dist(x1+x2)<0.5, else 0
    # limits = [0.5, 0.25, 0.1]
    limits = [0.25]
    for i, lim in enumerate(limits):
        logging.info(f'all if less {lim} statistics')
        ind_nonzero = np.where(dist_x < lim)[0]
        c_array2 = c_array[ind_nonzero]
        dist = dist_all[ind_nonzero]
        print('nonzero: ', len(ind_nonzero))
        output_by_percent(c_array2, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
                          os.path.join(path['output'], f'postprocess_all_if_less_{lim}'), i_stat=5 + i, mirror=mirror)
        np.savez(os.path.join(path['output'], f'postprocess_all_if_less_{lim}', 'dist_data.npz'), dist=dist)
    ###################################################################
    # # cp statistics
    # logging.info('cp statistics')
    # dist = dist_by_sumstat(sumstat_all[:, :Truth.length[0]], Truth.cp[:, 1])
    # output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_cp'), i_stat=0, mirror=mirror)
    # np.savez(os.path.join(path['output'], 'postprocess_cp', 'dist_data.npz'), dist=dist)
    # ###################################################################
    # # u statistics
    # logging.info('U statistics')
    # dist = dist_by_sumstat(sumstat_all[:, Truth.length[0]:Truth.length[1]], Truth.u_flat[:, 0])
    # output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_u'), i_stat=1, mirror=mirror)
    # np.savez(os.path.join(path['output'], 'postprocess_u', 'dist_data.npz'), dist=dist)
    # ###################################################################
    # # uv statistics
    # logging.info('uv statistics')
    # dist = dist_by_sumstat(sumstat_all[:, Truth.length[1]:Truth.length[2]], -Truth.uv_flat[:, 0])
    # output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_uv'), i_stat=2, mirror=mirror)
    # np.savez(os.path.join(path['output'], 'postprocess_uv', 'dist_data.npz'), dist=dist)
    # ###################################################################
    # # cp + u statistics
    # logging.info('cp + U statistics')
    # dist = dist_by_sumstat(sumstat_all[:, :Truth.length[1]] / norm[:Truth.length[1]], sumstat_true[:Truth.length[1]])
    # output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_cp_u'), i_stat=3, mirror=mirror)
    # np.savez(os.path.join(path['output'], 'postprocess_cp_u', 'dist_data.npz'), dist=dist)
    ###################################################################
    # match inflow statistics
    logging.info('\n Match inflow statistics')
    output_folder = os.path.join(path['output'], 'postprocess_inflow')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # sumstat_all = sumstat_all[ind_nonzero]
    u0_ind = np.arange(Truth.length[0], Truth.length[0] + len(Truth.u[0]))
    uv0_ind = np.arange(Truth.length[1], Truth.length[1] + len(Truth.uv[0]))
    ind_first_profiles = np.concatenate((u0_ind, uv0_ind))
    norm_u = [np.max(sumstat_true[u0_ind])]*len(Truth.u[0])
    norm_uv = [np.max(sumstat_true[uv0_ind])]*len(Truth.uv[0])
    norm_first_profile = np.concatenate((norm_u, norm_uv))

    dist_inflow = dist_by_sumstat(sumstat_all[:, ind_first_profiles] / norm_first_profile, sumstat_true[ind_first_profiles])

    ind_nonzero_inflow = np.where(dist_inflow < 36.5)[0]
    ind_nonzero_x = np.where(dist_x < 0.25)[0]
    plot_dist_pdf(output_folder+'/dist_inflow/', dist_inflow[ind_nonzero_inflow], 0.3)
    ind_nonzero_both = np.where(np.logical_and(dist_inflow < 36.5, dist_x < 0.25))[0]
    print('len(nonzero))', len(ind_nonzero_both))
    c_array3 = c_array[ind_nonzero_both]
    dist_inflow = dist_all[ind_nonzero_both]
    ind_sort_dist = np.argsort(dist_inflow)
    np.savez(os.path.join(path['output'], 'postprocess_inflow', 'ind_nonzero.npz'), ind_x=ind_nonzero_x,
             ind_inflow=ind_nonzero_inflow, ind_both=ind_nonzero_both, ind_sort_dist=ind_sort_dist)
    output_by_percent(c_array3, dist_inflow, C_limits, x_list, num_bin_raw, num_bin_kde,
                      os.path.join(path['output'], 'postprocess_inflow'), i_stat=20, mirror=mirror)
    np.savez(os.path.join(path['output'], 'postprocess_inflow', 'dist_data.npz'), dist=dist_inflow)
    ###################################################################

    # # x1 statistics
    # dist = dist_by_sumstat(sumstat_all[:, Truth.length[2]], sumstat_true[Truth.length[2]])
    # output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_x1'), i_stat=15, mirror=mirror)
    # # x2 statistics
    # dist = dist_by_sumstat(sumstat_all[:, Truth.length[2]+1], sumstat_true[Truth.length[2]+1])
    # output_by_percent(c_array, dist, C_limits, x_list, num_bin_raw, num_bin_kde,
    #                   os.path.join(path['output'], 'postprocess_x2'), i_stat=16, mirror=mirror)
    #


if __name__ == '__main__':
    main()
