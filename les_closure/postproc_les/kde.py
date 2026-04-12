from scipy.stats import gaussian_kde
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut
import numpy as np


from abc_code.utils import timer
from time import time


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
        Xgrid, Ygrid= np.meshgrid(xgrid, ygrid, indexing='ij')
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
    return Z, C_max, ind[0]


# def gaussian_kde_sklearn(data, a, b, num_bin_joint):
#     dim = len(a)
#     C_max = []
#
#     time1 = time()
#     bandwidths = np.linspace(0.05, 0.15, 30)
#     grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=4, verbose=10, n_jobs=4)
#     grid.fit(data)
#     bw = grid.best_params_
#     print('bw = ', bw)
#     time2 = time()
#     timer(time1, time2, "Time for Cross-validation")
#     # exit()
#
#     time1 = time()
#     print('bw = ', bw)
#     kde = KernelDensity(bandwidth=bw, kernel='gaussian')
#     kde.fit(data)
#     print(data.shape)
#     # # evaluate on a regular grid
#     xgrid = np.linspace(a[0], b[0], num_bin_joint + 1)
#     ygrid = np.linspace(a[1], b[1], num_bin_joint + 1)
#     zgrid = np.linspace(a[2], b[2], num_bin_joint + 1)
#     if dim == 3:
#         Xgrid, Ygrid, Zgrid = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
#         Z = np.exp(kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel(), Zgrid.ravel()]).T))
#         Z = Z.reshape(Xgrid.shape)
#         ind = np.argwhere(Z == np.max(Z))
#         for i in ind:
#             C_max.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]]])
#     elif dim == 4:
#         z4grid = np.linspace(a[3], b[3], num_bin_joint + 1)
#         Xgrid, Ygrid, Zgrid, Z4grid = np.meshgrid(xgrid, ygrid, zgrid, z4grid, indexing='ij')
#         Z = np.exp(kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel(), Zgrid.ravel(), Z4grid.ravel()])))
#         Z = Z.reshape(Xgrid.shape)
#         ind = np.argwhere(Z == np.max(Z))
#         for i in ind:
#             C_max.append([xgrid[i[0]], ygrid[i[1]], zgrid[i[2]], z4grid[i[3]]])
#     time2 = time()
#     timer(time1, time2, "Time for gaussian_kde_scilearn")
#     return Z, C_max