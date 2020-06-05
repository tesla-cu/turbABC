import logging
import numpy as np
import os
import itertools
from pyabc.kde import gaussian_kde_scipy, bw_from_kdescipy
from pyabc.utils import pdf_from_array_with_x, define_eps


def calc_raw_joint_pdf(accepted, num_bin_joint, C_limits, weights=None):

    N_params = len(C_limits)
    # C_final_joint
    C_final_joint = []
    H, edges = np.histogramdd(accepted, bins=num_bin_joint, range=C_limits, weights=weights)
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
    if len(ind) > 10:
        logging.warning('Can not estimate parameters from joint pdf!'
                        'Too many bins ({} bins, max value {}) '
                        'with the same max value in joint pdf'.format(len(ind), np.max(H)))
    else:
        logging.info('Estimated parameters from joint pdf: {}'.format(C_final_joint))
    return H, C_final_joint


def calc_conditional_pdf_smooth(Z, data_folder):

    N_params = len(Z.shape)
    params = np.arange(N_params)
    ind_max = np.argwhere(Z == np.max(Z))[0]

    for i in range(N_params):
        for j in range(N_params):
            if i > j:
                # Smooth
                index = tuple(np.where(np.logical_and(params != i, params != j))[0])
                H = Z.copy()
                for k, ind in enumerate(index):
                    H = np.take(H, ind_max[ind], axis=ind-k)
                np.savetxt(os.path.join(data_folder, 'conditional_smooth{}{}'.format(i, j)), H.T)


def calc_marginal_pdf_smooth(Z, num_bin_joint, C_limits, data_folder):

    N_params = len(C_limits)
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                ind = tuple(np.where(np.arange(N_params) != i)[0])
                y = np.sum(Z, axis=ind)
                x = np.linspace(C_limits[i, 0], C_limits[i, 1], num_bin_joint + 1)
                y = y/np.sum(y)
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
                np.savetxt(os.path.join(path, 'marginal_{}'.format(i)), [x, y])
            elif i < j:
                # note: the histogram does not follow the Cartesian convention x values are on the abscissa and y values
                # on the ordinate axis. Rather, x is histogrammed along the first dimension of the array (vertical),
                # and y along the second dimension of the array (horizontal)
                H, xedges, yedges = np.histogram2d(x=accepted[:, i], y=accepted[:, j], bins=num_bin_joint,
                                                   range=[C_limits[i], C_limits[j]], density=1)
                np.savetxt(os.path.join(path, 'marginal{}{}'.format(i, j)), H)
                np.savetxt(os.path.join(path, 'marginal_bins{}{}' .format(i, j)), [xedges, yedges])


def marginal_confidence(N_params, path, level):

    confidence = np.zeros((N_params, 2))
    for i in range(N_params):
        data_marg = np.loadtxt(os.path.join(path, 'marginal_smooth{}'.format(i)))
        x, y = data_marg[0], data_marg[1]
        dx = x[1] - x[0]
        y = y/(np.sum(y)*dx)    # normalized
        cdf = np.zeros_like(y)
        for j in range(len(y)):
            cdf[j] = np.sum(y[:j+1]*dx)
        confidence[i, 0] = np.interp([level], cdf, x)
        confidence[i, 1] = np.interp([1-level], cdf, x)
    np.savetxt(os.path.join(path, 'confidence_{}'.format(int(100*(1-level)))), confidence)


def marginal_confidence_joint(accepted, path, level):

    N_params = len(accepted[0])
    confidence = np.zeros((N_params, 2))
    for i in range(N_params):
        # accepted_tmp = np.sort(accepted[:, i])
        accepted_tmp = accepted[:, i]
        confidence[i, 0] = np.percentile(accepted_tmp, int(100*level))
        confidence[i, 1] = np.percentile(accepted_tmp, int(100*(1-level)))
    np.savetxt(os.path.join(path, 'quantile_{}'.format(int((1-level)*100))), confidence)


def mirror_data_for_kde(points, left, right, weights=None):

    # checking the equation of ellipse with the given point
    inside_ellipse = lambda point, corner: 0 < np.sum(np.power(point - corner, 2) / np.power(size, 2)) < 1

    def get_corners(left, right):
        n_params = len(left)
        limits = list(zip(left, right))
        indices = list(itertools.product(range(2), repeat=n_params))
        corners = [[limits[i][ind] for (i, ind) in enumerate(corner)] for corner in indices]
        return corners

    k = 1  # coefficient for reflection interval
    size = k * bw_from_kdescipy(points)
    print('percent of interval to reflect', size / (np.array(right) - np.array(left)))

    new_points = list(points)
    if weights:
        new_weights = list(weights)
    else:
        weights = [1]*len(new_points)
        new_weights = weights
    corners = get_corners(left, right)
    for point, w in zip(points, weights):
        # axis reflection:
        for i, p in enumerate(point):
            for lim in [left[i], right[i]]:
                if 0 < np.abs(p - lim) < size[i]:
                    point_new = point
                    point_new[i] = 2*lim - p
                    new_points.append(point_new)
                    new_weights.append(w)
        # point reflection
        for corner in corners:
            if inside_ellipse(point, corner):
                point_new = 2 * corner - point
                new_points.append(point_new)
                new_weights.append(w)
    return np.array(new_points), np.array(new_weights)


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




