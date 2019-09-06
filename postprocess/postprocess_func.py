import logging
import numpy as np
import os
from pyabc.utils import gaussian_kde_scipy
from pyabc.utils import pdf_from_array_with_x, define_eps


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
    np.savetxt(os.path.join(path, 'C_final_joint{}'.format(num_bin_joint)), C_final_joint)
    if len(ind) > 10:
        logging.warning('Can not estimate parameters from joint pdf!'
                        'Too many bins ({} bins, max value {}) '
                        'with the same max value in joint pdf'.format(len(ind), np.max(H)))
    else:
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_joint))
    #
    # # Gaussian smoothness
    Z, C_final_smooth = gaussian_kde_scipy(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_joint)
    np.savetxt(os.path.join(path, 'C_final_smooth'+str(num_bin_joint)), C_final_smooth)
    logging.info('Estimated parameters from joint pdf: {}'.format(C_final_smooth))
    return Z, C_final_smooth
####################################################################################################################


def calc_marginal_pdf_smooth(Z, num_bin_joint, C_limits, data_folder):

    N_params = len(C_limits)
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                y = np.sum(Z, axis=tuple(np.where(np.arange(N_params) != i)[0]))
                x = np.linspace(C_limits[i, 0], C_limits[i, 1], num_bin_joint + 1)
                np.savetxt(os.path.join(data_folder, 'marginal_smooth{}'.format(i)), [x, y])
            elif i < j:
                # Smooth
                params = np.arange(N_params)
                ind = tuple(np.where(np.logical_and(params != i, params != j))[0])
                H = np.sum(Z, axis=ind)
                np.savetxt(os.path.join(data_folder, 'marginal_smooth{}{}'.format(i, j)), H)


def calc_marginal_pdf_raw(accepted, num_bin_joint, C_limits, path):
    N_params = len(C_limits)
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                x, y = pdf_from_array_with_x(accepted[:, i], bins=num_bin_joint, range=C_limits[i])
                np.savetxt(os.path.join(path['output'], 'marginal_{}'.format(i)), [x, y])
            elif i < j:
                H, xedges, yedges = np.histogram2d(x=accepted[:, j], y=accepted[:, i], bins=num_bin_joint,
                                                   range=[C_limits[j], C_limits[i]])
                print(i, j, H)
                np.savetxt(os.path.join(path['output'], 'marginal{}{}'.format(i, j)), H)
                np.savetxt(os.path.join(path['output'], 'marginal_bins{}{}' .format(i, j)), [xedges, yedges])


# def bootstrapping(Z, C_final, C_limits, n_sample):
#
#     basefolder = './ABC/without_noise/40_bigger_domain/'
#     path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots/')}
#     num_bin_joint = Z.shape[0]
#     xgrid = np.linspace(C_limits[0, 0], C_limits[0, 1], num_bin_joint)
#     ygrid = np.linspace(C_limits[1, 0], C_limits[1, 1], num_bin_joint)
#     zgrid = np.linspace(C_limits[2, 0], C_limits[2, 1], num_bin_joint)
#     z4grid = np.linspace(C_limits[3, 0], C_limits[3, 1], num_bin_joint)
#     Z_flat = Z.flatten()
#     C_max = []
#     for i in range(n_sample):
#         sample = np.random.choice(a=Z_flat, size=len(Z), replace=True)
#         ind = np.argwhere(Z == np.max(sample))
#         for i in ind:
#             C_max.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]], z4grid[i[3]]])
#
#     confidence = np.zeros((len(C_limits), 2))
#     for i in range(len(C_limits)):
#         confidence[i] = plotting.plot_bootstrapping_pdf(path, np.array(C_max)[:, i], 2.5, i, C_limits[i], C_final[i])
#
#     print(confidence)


def marginal_confidence(N_params, path, level):

    confidence = np.zeros((N_params, 2))
    for i in range(4):
        data_marg = np.loadtxt(os.path.join(path, 'marginal_smooth{}'.format(i)))
        x, y = data_marg[0], data_marg[1]
        dx = x[1] - x[0]
        y = y/(np.sum(y)*dx)    # normalized
        cdf = np.zeros_like(y)
        for j in range(len(y)):
            cdf[j] = np.sum(y[:j+1]*dx)
        # print(cdf)
        confidence[i, 0] = np.interp([level], cdf, x)
        confidence[i, 1] = np.interp([1-level], cdf, x)
    np.savetxt(os.path.join(path, 'confidence'), confidence)


def main():

    basefolder = './'

    path = {'output': os.path.join(basefolder, 'output/'),
            'plots': os.path.join(basefolder, 'plots/')}
    path['calibration'] = os.path.join(path['output'], 'calibration/')

    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n#############POSTPROCESSING############')
    ####################################################################################################################
    ### For classic ABC ########
    ####################################################################################################################
    N_files = 1
    N_params = 4
    C_limits = np.loadtxt(os.path.join(path['calibration'], 'C_limits'))
    files = [os.path.join(path['output'], 'classic_abc{}.npz'.format(i)) for i in range(N_files)]
    accepted = np.empty((0, N_params))
    dist = np.empty((0, 1))
    for file in files:
        accepted = np.vstack((accepted, np.load(file)['C']))
        dist = np.vstack((dist, np.load(file)['dist'].reshape((-1, 1))))
    data = np.hstack((accepted, dist)).tolist()
    for x in [0.03, 0.05, 0.1, 0.3, 0.5]:
        folder = os.path.join(path['output'], 'x_0_{}'.format(str(x)[2:]))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        eps = define_eps(data, x)
        abc_accepted = accepted[np.where(dist < eps)[0]]

        num_bin_joint = 20
        Z, C_final_smooth = calc_final_C(abc_accepted, num_bin_joint, C_limits, folder)
        np.savez(os.path.join(folder, 'Z.npz'), Z=Z)
        calc_marginal_pdf_smooth(Z, num_bin_joint, C_limits, folder)
        marginal_confidence(N_params, folder, 0.05)

    # ####################################################################################################################
    # ### For IMCMC ########
    # ####################################################################################################################
    # logging.info('\n#############Calibration############')
    # calibration1_c = np.load(os.path.join(path['calibration'], 'calibration1.npz'))['C']
    # calibration1_dist = np.load(os.path.join(path['calibration'], 'calibration.npz'))['dist']
    # calibration2_c = np.load(os.path.join(path['calibration'], 'calibration1.npz'))['C']
    # calibration2_dist = np.load(os.path.join(path['calibration'], 'calibration.npz'))['dist']
    # eps1 = np.loadtxt(os.path.join(path['calibration'], 'eps1'))
    # eps2 = np.loadtxt(os.path.join(path['calibration'], 'eps2'))
    # C_limits = np.loadtxt(os.path.join(path['calibration'], 'C_limits'))
    # logging.info('1: min dist = {}'.format(np.min(calibration1_dist)))
    # logging.info('1: eps = {}'.format(eps1))
    # logging.info('2: min dist = {}'.format(np.min(calibration2_dist)))
    # logging.info('2: eps = {}'.format(eps2))
    # # ##################################################################################################################
    # # # logging.info('noise = {}'.format((eps_k-min_dist)*0.03))
    # # accepted = calibration[np.where(dist < eps_k)[0]]
    # # logging.info('accepted {}% ({}/{})'.format(np.round((accepted.shape[0]/calibration.shape[0])*100, 2),
    # #                                            accepted.shape[0], calibration.shape[0]))
    # # if accepted.shape[0] == 0:
    # #     print("There is no accepted parametes, consider increasing eps.")
    # #     exit()
    # # np.savez(os.path.join(path['output'], 'accepted.npz'), C=accepted)
    #
    # prior = np.load(os.path.join(path['calibration'], 'prior.npz'))['Z']
    # calc_marginal_pdf_smooth(prior, 10, C_limits, path['calibration'])
    #
    # ####################################################################################################################
    # ### Adaptive MCMC ########
    # ####################################################################################################################
    # N_chains = 10
    # N_files = 1
    # chain_files = [os.path.join(path['output'], 'chain{}_{}'.format(i, j)) for i in range(N_chains) for j in range(N_files)]
    # accepted = np.empty(N, )
    # for file in chain_files:
    #     accepted = np.load(os.path.join(path['output'], 'calibration1.npz'))['C']
    #
    # Z, C_final_smooth = calc_final_C(accepted, num_bin_joint, C_limits, path)
    # calc_marginal_pdf_smooth(Z, num_bin_joint, C_limits, path)
    # # np.savez(os.path.join(path['output'], 'Z.npz'), Z=Z)
    # # marginal_confidence(path, 0.05)

    logging.info('\n#############Done############')


if __name__ == '__main__':
    main()