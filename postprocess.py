import logging
import numpy as np
import os
from scipy.stats import gaussian_kde

from utils import timer
from time import time
import plotting




def calc_final_C(accepted, num_bin_joint, C_limits, path):
    """ Estimate the best fit of parameters based on joint pdf.
    """

    N_params = len(C_limits)

    # C_final_joint
    C_final_joint = []
    H, edges = np.histogramdd(accepted, bins=num_bin_joint, range=C_limits)
    logging.debug('Max number in bin: {}'.format(np.max(H)))
    logging.debug('Mean number in bin: {}'.format(np.mean(H)))
    edges = np.array(edges)
    C_bin = (edges[:, :-1] + edges[:, 1:]) / 2  # shift value in the center of the bin
    ind = np.argwhere(H == np.max(H))
    for i in ind:
        point = []
        for j in range(N_params):
            point.append(C_bin[j, i[j]])
        C_final_joint.append(point)
    np.savetxt(os.path.join(path['output'], 'C_final_joint' + str(num_bin_joint)), C_final_joint)
    if len(ind) > 10:
        logging.warning('Can not estimate parameters from joint pdf!'
                        'Too many bins ({} bins, max value {}) '
                        'with the same max value in joint pdf'.format(len(ind), np.max(H)))
    else:
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_joint))
    #
    # # Gaussian smoothness
    Z, C_final_smooth = gaussian_kde_scipy(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_joint)

    # np.savetxt(os.path.join(path['output'], 'C_final_smooth'+str(self.num_bin_joint)), self.C_final_smooth)
    np.savetxt(os.path.join(path['output'], 'C_final_smooth'), C_final_smooth)
    # np.savetxt(os.path.join(path['output'], 'posterior' + str(self.num_bin_joint)), Z)
    logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
    return Z, C_final_smooth
####################################################################################################################


def calc_marginal_pdf(Z, num_bin_joint, C_limits, path):

    N_params = len(C_limits)
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                y = np.sum(Z, axis=tuple(np.where(np.arange(N_params) != i)[0]))
                x = np.linspace(C_limits[i, 0], C_limits[i, 1], num_bin_joint + 1)
                np.savetxt(os.path.join(path['output'], 'marginal_smooth{}'.format(i)), [x, y])
            elif i < j:
                # Smooth
                params = np.arange(N_params)
                ind = tuple(np.where(np.logical_and(params != i, params != j))[0])
                H = np.sum(Z, axis=ind)
                np.savetxt(os.path.join(path['output'], 'marginal_smooth{}{}'.format(i, j)), H)


def gaussian_kde_scipy(data, a, b, num_bin_joint):
    dim = len(a)
    C_max = []
    print(dim, data.shape, a, b)
    data_std = np.std(data, axis=0)
    kde = gaussian_kde(data.T, bw_method='scott')
    f = kde.covariance_factor()
    bw = f * data_std
    print('Scott: f, bw = ', f, bw)
    # kde = gaussian_kde(data.T, bw_method='silverman')
    # f = kde.covariance_factor()
    # bw = f * data_std
    # print('Silverman: f, bw = ', f, bw)
    # kde.set_bandwidth(bw_method=kde.factor / 4.)
    # f = kde.covariance_factor()
    # bw = f * data_std
    # print('f, bw = ', f, bw)

    time1 = time()
    # # evaluate on a regular grid
    xgrid = np.linspace(a[0], b[0], num_bin_joint + 1)
    if dim == 1:
        Z = kde.evaluate(xgrid)
        Z = Z.reshape(xgrid.shape)
        ind = np.argwhere(Z == np.max(Z))
        for i in ind:
            C_max.append(xgrid[i])
    elif dim == 2:
        ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='ij')
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z = Z.reshape(Xgrid.shape)
    elif dim == 3:
        ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
        zgrid = np.linspace(a[2], b[2], num_bin_joint + 1)
        Xgrid, Ygrid, Zgrid = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel(), Zgrid.ravel()]))
        Z = Z.reshape(Xgrid.shape)
        ind = np.argwhere(Z == np.max(Z))
        for i in ind:
            C_max.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]]])
    elif dim == 4:
        ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
        zgrid = np.linspace(a[2], b[2], num_bin_joint + 1)
        z4grid = np.linspace(a[3], b[3], num_bin_joint + 1)
        Xgrid, Ygrid, Zgrid, Z4grid = np.meshgrid(xgrid, ygrid, zgrid, z4grid, indexing='ij')
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel(), Zgrid.ravel(), Z4grid.ravel()]))
        Z = Z.reshape(Xgrid.shape)
        ind = np.argwhere(Z == np.max(Z))
        for i in ind:
            C_max.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]], z4grid[i[3]]])
    else:
        print("gaussian_kde_scipy: Wrong number of dimensions (dim)")
    time2 = time()
    timer(time1, time2, "Time for gaussian_kde_scipy")
    return Z, C_max


def main():

    basefolder = './ABC/without_noise/40_biger_range/'

    path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots/')}
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABC_log_post')), logging.StreamHandler()],
        level=logging.DEBUG)

    calibration = np.load(os.path.join(path['output'], 'calibration.npz'))['C']
    dist = np.load(os.path.join(path['output'], 'calibration.npz'))['dist']
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits'))

    eps_k = plotting.plot_dist_pdf(path, dist, 0.05)
    ####################################################################################################################
    min_dist = np.min(dist)
    logging.info('min dist = {}'.format(min_dist))
    logging.info('eps = {}'.format(eps_k))
    # logging.info('noise = {}'.format((eps_k-min_dist)*0.03))
    accepted = calibration[np.where(dist < eps_k)[0]]
    logging.info('accepted {}% ({}/{})'.format(np.round((accepted.shape[0]/calibration.shape[0])*100, 2),
                                               accepted.shape[0], calibration.shape[0]))
    if accepted.shape[0] == 0:
        print("There is no accepted parametes, consider increasing eps.")
        exit()
    np.savez(os.path.join(path['output'], 'accepted.npz'), C=accepted)
    ####################################################################################################################
    num_bin_joint = 20
    Z, C_final_smooth = calc_final_C(accepted, num_bin_joint, C_limits, path)
    calc_marginal_pdf(Z, num_bin_joint, C_limits, path)


if __name__ == '__main__':
    main()