import numpy as np
import os

import postprocess.postprocess_func as pp
from pyabc.kde import gaussian_kde_scipy, find_MAP_kde, grid_for_kde
from scipy.stats import gaussian_kde


def calc_bounds_for_mirror(points):
    N_params = len(points[0])
    left_bound, right_bound = [], []
    for p in range(N_params):
        bin_centers = np.unique(points[:, p])
        left_bound.append(np.min(bin_centers))
        right_bound.append(np.max(bin_centers))
    return left_bound, right_bound


def make_3d_pdf(accepted, num_bin_raw, C_limits, bound, grid_kde, output_folder, dist_weight=None):

    if dist_weight is None:
        unique, weights = np.unique(accepted, axis=0, return_counts=True)
    else:
        unique, weights = accepted, dist_weight
    np.savetxt(os.path.join(output_folder, 'unique.txt'), unique)
    np.savetxt(os.path.join(output_folder, 'weights.txt'), weights)
    print("Number of unique samples (without noise added) is", len(unique))
    print(f'Std of the unique data is {np.std(unique, axis=0)}')
    # # # ##############################################################################
    # # 3d histogram
    print(f'2D raw marginals with {num_bin_raw} bins per dimension')
    pp.calc_raw_joint_pdf(unique, num_bin_raw, C_limits, weights)
    pp.calc_marginal_pdf_raw(unique, num_bin_raw, C_limits, output_folder, weights)
    # # # ##############################################################################
    print(f'2D smooth marginals with {grid_kde} bins per dimension')
    (left, right) = bound
    # # 3d kde
    Z = gaussian_kde_scipy(unique, left, right, grid_kde, weights=weights)
    np.savez(os.path.join(output_folder, 'Z.npz'), Z=Z)
    map_value = find_MAP_kde(Z, left, right)
    np.savetxt(os.path.join(output_folder, 'map_value'), map_value)
    pp.calc_marginal_pdf_smooth(Z, grid_kde, bound, output_folder)
    # 3d mirror
    unique, weights = kde_2d_3d_mirror_data(unique, weights, left, right)
    Z = gaussian_kde_scipy(unique, left, right, grid_kde, weights=weights)
    pp.calc_marginal_pdf_smooth(Z, grid_kde, bound, output_folder, name='_mirror')
    pp.calc_conditional_pdf_smooth(Z, output_folder, name='_mirror')
    np.savez(os.path.join(output_folder, 'Z_mirror.npz'), Z=Z)
    map_value = find_MAP_kde(Z, left, right)
    np.savetxt(os.path.join(output_folder, 'map_value'), map_value)


#############################################################################
# Weighting by distance
#############################################################################
def make_3d_pdf_weight(accepted, weights, num_bin_raw, C_limits, grid_kde, output_folder):

    left, right = C_limits[:, 0], C_limits[:, 1]
    np.savetxt(os.path.join(output_folder, 'unique.txt'), accepted)
    np.savetxt(os.path.join(output_folder, 'weights.txt'), weights)
    print(f'Std of the accepted data is {np.std(accepted, axis=0)}')
    # # # ##############################################################################
    # # 3d histogram
    print(f'2D raw marginals with {num_bin_raw} bins per dimension')
    pp.calc_raw_joint_pdf(accepted, num_bin_raw, C_limits, weights)
    pp.calc_marginal_pdf_raw(accepted, num_bin_raw, C_limits, output_folder, weights)
    # # # ##############################################################################
    # # 3d kde
    print(f'2D smooth marginals with {grid_kde} bins per dimension')
    Z = gaussian_kde_scipy(accepted, C_limits[:, 0], C_limits[:, 1], grid_kde, weights=weights)
    np.savez(os.path.join(output_folder, 'Z.npz'), Z=Z)
    map_value = find_MAP_kde(Z, left, right)
    np.savetxt(os.path.join(output_folder, 'map_value'), map_value)
    pp.calc_marginal_pdf_smooth(Z, grid_kde, C_limits, output_folder, name='_dist')


def make_2d_pdf(accepted, bounds, grid_kde, output_folder):
    #############################################################################
    # 2d calculate weights from duplicates
    #############################################################################
    left, right = np.array(bounds[0]), np.array(bounds[1])
    for p in range(3):
        for j in range(3):
            if p < j:
                left_2d, right_2d = left[[p, j]], right[[p, j]]
                unique_2d, weights_2d = np.unique(accepted[:, (p, j)], axis=0, return_counts=True)
                # Smooth
                Z = gaussian_kde_scipy(unique_2d, left_2d, right_2d, grid_kde, weights=weights_2d)
                np.savetxt(os.path.join(output_folder, f'marginal_smooth_2d{p}{j}'), Z)
                # 2d mirror
                unique_2d, weights_2d = kde_2d_3d_mirror_data(unique_2d, weights_2d, left_2d, right_2d)
                Z = gaussian_kde_scipy(unique_2d, left_2d, right_2d, grid_kde, weights=weights_2d)
                np.savetxt(os.path.join(output_folder, f'marginal_smooth_2d_mirror{p}{j}'), Z)


def kde_2d_3d_mirror_data(points, weights, left, right):

    # checking the equation of ellipse with the given point
    inside_ellipse = lambda point, corner: 0 < np.sum(np.power(point - corner, 2) / np.power(size, 2)) < 1

    def get_corners(left, right):
        n_params = len(left)
        limits = np.array([[l, r] for (l, r) in zip(left, right)])
        if n_params == 3:
            return np.array([np.array([x, y, z]) for x in limits[0] for y in limits[1] for z in limits[2]])
        elif n_params == 2:
            return np.array([np.array([x, y]) for x in limits[0] for y in limits[1]])

    k = 1  # coefficient for reflection interval
    kde = gaussian_kde(points.T, weights=weights)
    f = kde.covariance_factor()
    size = k * f * np.std(points, axis=0)
    print('percent of interval to reflect', size / (np.array(right) - np.array(left)))

    new_points = list(points)
    new_weights = list(weights)
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


def make_1d_pdf(accepted, num_bin_raw, C_limits, grid_kde, output_folder):

    N_params = len(C_limits)
    if not hasattr(num_bin_raw, "__len__"):
        num_bin_raw = [num_bin_raw]*N_params
    # left, right = calc_bounds_for_mirror(accepted)
    left = C_limits[:, 0] + (C_limits[:, 1] - C_limits[:, 0])/40
    right = C_limits[:, 1] - (C_limits[:, 1] - C_limits[:, 0])/40
    MAP_hist = []
    for p in range(N_params):
        unique_1d_p, counts_p = np.unique(accepted[:, p], return_counts=True)
        MAP_ind = np.argmax(counts_p)
        MAP_hist.append(unique_1d_p[MAP_ind])
        print(f"unique points in 1D: {unique_1d_p}")
    # # 1d histogram
        hist, edges = np.histogram(unique_1d_p, bins=num_bin_raw[p], density=1, weights=counts_p,
                                   range=[C_limits[p, 0], C_limits[p, 1]])
        np.savetxt(os.path.join(output_folder, f'marginal_hist_1d{p}'), hist)
        np.savetxt(os.path.join(output_folder, f'hist_bins_1d{p}'), edges)
    # # 1d kde
        Z = gaussian_kde_scipy(unique_1d_p, left[p], right[p], grid_kde, weights=counts_p)
        kde = gaussian_kde(unique_1d_p.T, weights=counts_p)
        grid, _ = grid_for_kde(left[p], right[p], grid_kde)
        data2save = np.vstack((np.array(grid)[0], Z))
        np.savetxt(os.path.join(output_folder, f'marginal_smooth_1d{p}'), data2save)

        f = kde.covariance_factor()
        size_reflection = f*np.std(unique_1d_p)
        # # 1d kde mirrored
        if MAP_ind > num_bin_raw[p]/3:
            if MAP_ind != num_bin_raw[p] - 1:
                Z, grid = special_mirror(unique_1d_p, counts_p, grid_kde, (left[p], right[p]), map_hist=MAP_hist[p], side='r')
            else:
                size_reflection *= len(unique_1d_p) * counts_p[-1] / np.max(counts_p)
                Z, grid = mirror_1d_kde(unique_1d_p, counts_p, grid_kde, (left[p], right[p]), side='r', size=size_reflection)
        elif MAP_ind < num_bin_raw[p]/3 - 1:
            if MAP_ind != 0:
                Z, grid = special_mirror(unique_1d_p, counts_p, grid_kde, (left[p], right[p]), map_hist=MAP_hist[p], side='l')
            else:
                size_reflection *= len(unique_1d_p) * counts_p[0] / np.max(counts_p)
                Z, grid = mirror_1d_kde(unique_1d_p, counts_p, grid_kde, (left[p], right[p]), side='l', size=size_reflection)
        data2save = np.vstack((np.array(grid)[0], Z))
        np.savetxt(os.path.join(output_folder, f'marginal_smooth_1d_mirror{p}'), data2save)
    np.savetxt(os.path.join(output_folder, 'MAP_hist_1d'), MAP_hist)


def special_mirror(unique_1d, weights_1d, grid_kde, bound, map_hist, side):

    n_points = len(unique_1d)
    if side == 'r':
        reflection_point = bound[1]
        size_ind = np.where(unique_1d < map_hist - (reflection_point - map_hist))
    if side == 'l':
        reflection_point = bound[0]
        size_ind = np.where(unique_1d > map_hist + (map_hist - reflection_point))
    counts_p = np.concatenate((weights_1d, weights_1d[size_ind]))
    unique_1d_p = np.concatenate((unique_1d, 2 * map_hist - unique_1d[size_ind]))  # a - (x-a) or b + (b-x)

    Z = gaussian_kde_scipy(unique_1d_p, bound[0], bound[1], 2 * grid_kde, weights=counts_p)
    grid, _ = grid_for_kde(bound[0], bound[1], 2 * grid_kde)
    norm = len(unique_1d_p) / n_points

    return Z * norm, grid


def mirror_1d_kde(unique_1d, weights_1d, grid_kde, C_limits, side, size=None):

    n_points = len(unique_1d)
    if 'r' in side:
        reflection_point = C_limits[1]
        if size:
            size_ind = np.where((reflection_point > unique_1d) & (unique_1d > reflection_point - size))
        else:
            size_ind = np.arange(int(n_points / 2), n_points)
    elif side == 'l':
        reflection_point = C_limits[0]
        if size:
            size_ind = np.where((reflection_point < unique_1d) & (unique_1d < reflection_point + size))
        else:
            size_ind = np.arange(int(n_points / 2))

    counts_p = np.concatenate((weights_1d, weights_1d[size_ind]))
    unique_1d_p = np.concatenate((unique_1d, 2 * reflection_point - unique_1d[size_ind]))   # a - (x-a) or b + (b-x)
    Z = gaussian_kde_scipy(unique_1d_p, C_limits[0], C_limits[1], 2 * grid_kde, weights=counts_p)
    grid, _ = grid_for_kde(C_limits[0], C_limits[1], 2 * grid_kde)
    norm = len(unique_1d_p) / n_points

    return Z * norm, grid




