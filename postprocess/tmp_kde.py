import os
import numpy as np
import glob
from time import time
import pyabc.utils as utils
import pyabc.kde as my_kde
import postprocess.postprocess_func as pp
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import matplotlib.pyplot as plt
from KDEpy import FFTKDE

path_base = '../runs_abc/'
path = {'output': os.path.join(path_base, 'output'), 'plots': os.path.join(path_base, 'plots_kde')}
if not os.path.isdir(path['plots']):
    os.makedirs(path['plots'])

N_array = [100, 500, 1000, 2500, 5000, 7500, 10000]


def kde_scipy(data, grid_ravel, bandwidth=0.2, **kwargs):
    kde = gaussian_kde(data.T, bw_method='scott', **kwargs)
    return kde.evaluate(grid_ravel)


def kde_statsmodels_m(data, grid_ravel, bandwidth=0.2, **kwargs):
    kde = KDEMultivariate(data, bw='normal_reference', var_type='c'*len(data[0]), **kwargs)
    return kde.pdf(grid_ravel)


def kde_sklearn(data, grid_ravel, bandwidth=0.2, **kwargs):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', **kwargs)
    kde.fit(data)
    log_pdf = kde.score_samples(grid_ravel.T)
    return np.exp(log_pdf)


def kde_sklearn_kd_tree(data, grid_ravel, bandwidth=0.2, **kwargs):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='kd_tree', rtol=1e-4, **kwargs)
    kde.fit(data)
    log_pdf = kde.score_samples(grid_ravel.T)
    return np.exp(log_pdf)


kde_funcs = [kde_statsmodels_m, kde_scipy, kde_sklearn, kde_sklearn_kd_tree]
kde_funcnames = ['Statsmodels-M', 'Scipy', 'Sklearn', 'Sklearn: KD tree']

from scipy.stats.distributions import norm

# # The grid we'll use for plotting
# x_grid = np.linspace(-4.5, 3.5, 1000).reshape((1, -1))
#
# # Draw points from a bimodal distribution in 1D
# np.random.seed(0)
# x = np.concatenate([norm(-1, 1.).rvs(400), norm(1, 0.3).rvs(100)]).reshape((-1, 1))
# print(x.shape)
# pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))
# bw = my_kde.bw_from_kdescipy(x)[0]
# print(bw)
# #####################################################################
# fig = plt.figure()
# ax = plt.gca()
# for i in range(4):
#     print(kde_funcnames[i])
#     pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
#     ax.plot(x_grid[0], pdf.T, alpha=0.5, lw=3, label=kde_funcnames[i])
# ax.fill(x_grid[0], pdf_true[0], ec='gray', fc='gray', alpha=0.4)
# plt.legend()
# ax.set_xlim(-4.5, 3.5)
# fig.savefig(os.path.join(path['plots'], '1d_kde'))
# #####################################################################
# fig = plt.figure()
# ax = plt.gca()
# for bw_i in [0.1, bw, 0.7]:
#     pdf = kde_sklearn(x, x_grid, bandwidth=bw_i)
#     ax.plot(x_grid[0], pdf.T, alpha=0.9, lw=3, label='bw = {}'.format(np.round(bw_i, 3)))
# ax.fill(x_grid[0], pdf_true[0], ec='red', lw=3, fc='gray', alpha=0.7)
# ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.7, normed=True)
#
# plt.legend()
# ax.set_xlim(-4.5, 3.5)
# fig.savefig(os.path.join(path['plots'], '1d_kde_kernels'))


########################################################################################################################


# for N_i in N_array:
#         x = np.random.normal(size=N_i)
#         kwds['Scikit-learn']['rtol'] = rtol_i
#         for name, func in functions.items():
#             t = 0.0
#             for i in range(Nreps):
#                 t0 = time()
#                 func(x, xgrid, bw_i, **kwds[name])
#                 t1 = time()
#                 t += (t1 - t0)
#             times[name].append(t / Nreps)



def plot_marginal_smooth_pdf(data_folder, C_limits, plot_name):

    N_params = len(C_limits)
    max_value = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i)+str(j)] = np.loadtxt(os.path.join(data_folder, 'marginal_smooth{}{}'.format(i, j)))
                max_value = max(max_value, np.max(data[str(i)+str(j)]))
    max_value = int(max_value)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    # cmaplist[0] = 'black'   # force the first color entry to be black
    cmaplist[0] = 'white' # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    fig = plt.figure()
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(data_folder, 'marginal_smooth{}'.format(i)))
                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])

                im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
    fig.savefig(plot_name)
    plt.close('all')


C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
a, b = C_limits[:, 0], C_limits[:, 1]

# x_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3]
# x_list = [0.3]
# N_params = len(C_limits)
# files_abc = glob.glob1(path['output'], "classic_abc*.npz")
# files = [os.path.join(path['output'], i) for i in files_abc]
# accepted = np.empty((0, N_params))
# dist = np.empty((0, 1))
# print('Loading data')
# for file in files:
#     print('loading {}'.format(file))
#     accepted = np.vstack((accepted, np.load(file)['C']))
#     dist = np.vstack((dist, np.load(file)['dist'].reshape((-1, 1))))
# data = np.hstack((accepted, dist)).tolist()
# for x in x_list:
#     print('\n')
#     folder = os.path.join(path['output'], 'x_{}'.format(x * 100))
#     if not os.path.isdir(folder):
#         os.makedirs(folder)
#     eps = utils.define_eps(data, x)
#     np.savetxt(os.path.join(folder, 'eps'), [eps])
#     abc_accepted = accepted[np.where(dist < eps)[0]]
#     print('x = {}, eps = {}, N accepted = {} (total {})'.format(x, eps, len(abc_accepted), len(dist)))
#     num_bin_kde = 20
#     num_bin_raw = 20
# # accepted = normalize(accepted, axis=0, norm='max')
# # abc_accepted = abc_accepted[:N]
# np.savez(os.path.join(path['output'], 'tmp30'), accepted=abc_accepted)
########################################################################################################################













num_bin_joint = 20
params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']

data_load = np.load(os.path.join(path['output'], 'tmp30.npz'))['accepted']


#######################
# # 1D compare methods
# N = 1000
# ind = np.random.choice(np.arange(len(data_load)), size=N, replace=False)
# data = 2 * (data_load[ind] - a) / (b - a) - 1
# x = data[:, 0].reshape((-1, 1))
# x_grid = np.linspace(-1, 1, 20).reshape((1, -1))
# print(x.shape, x_grid.shape)
# bw = my_kde.bw_from_kdescipy(x)[0]
# print(bw)
# fig = plt.figure()
# ax = plt.gca()
# for i in range(4):
#     print(kde_funcnames[i])
#     pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
#     ax.plot(x_grid[0], pdf.T, alpha=0.5, lw=3, label=kde_funcnames[i])
# ax.hist(x, 20, fc='gray', histtype='stepfilled', alpha=0.7, normed=True)
# plt.legend()
# ax.set_xlim(-1, 1)
# fig.savefig(os.path.join(path['plots'], '1d_kde_realdata'))
#######################
# 1D compare speed
t = np.empty((len(kde_funcs), len(N_array)))
x_grid = np.linspace(-1, 1, 20).reshape((1, -1))
for n, N in enumerate(N_array):
    print('\nN = ', N)
    ind = np.random.choice(np.arange(len(data_load)), size=N, replace=False)
    data = 2 * (data_load[ind] - a) / (b - a) - 1
    x = data[:, 0].reshape((-1, 1))
    bw = my_kde.bw_from_kdescipy(x)[0]
    print(bw)
    for i in range(4):
        print(kde_funcnames[i])
        t0 = time()
        pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
        t1 = time()
        t[i, n] = (t1-t0)
    fig = plt.figure()
    ax = plt.gca()
    for i in range(4):
        ax.loglog(N_array, t[i], alpha=0.5, lw=3, label=kde_funcnames[i])
    plt.legend()
    fig.savefig(os.path.join(path['plots'], '1d_time_realdata'))
exit()
for N in N_array:
    print('\n###################################\nN = ', N)
    grid_mesh, grid_ravel = my_kde.grid_for_kde([-1]*4, [1]*4, num_bin_joint)
    bw = my_kde.bw_from_kdescipy(data)
    print('bw = ', bw)
    ########################################################################################################################
    print("\nscipy.Gaussian_kde")
    time1 = time()
    Z = kde_scipy(data.T, grid_ravel, 0.2)
    time2 = time()
    t1.append(time2 - time1)
    print("time = ", (time2-time1))
    # pp.calc_marginal_pdf_smooth(Z, num_bin_joint, C_limits, path['plots'])
    # plot_marginal_smooth_pdf(path['plots'], C_limits, os.path.join(path['plots'], 'scipy{}'.format(N)))
    ########################################################################################################################
    # Statsmodels
    print("\nStatsmodels")


    # Sklearn
    print("\nSklearn")
    time1 = time()
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
    kde.fit(data)
    Z = kde.score_samples(grid_ravel.T)
    Z = Z.reshape(grid_mesh[0].shape)
    time2 = time()
    t2.append(time2 - time1)
    print("time = ", (time2-time1))
    ########################################################################################################################
    print("\nSklearn: KD tree")
    time1 = time()
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian', algorithm='kd_tree',  rtol=1e-4)
    kde.fit(data)
    Z = kde.score_samples(grid_ravel.T)
    Z = Z.reshape(grid_mesh[0].shape)
    time2 = time()
    t3.append(time2 - time1)
    print("time = ", (time2-time1))
    ########################################################################################################################
    print("\nSklearn: Ball tree")
    time1 = time()
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian', algorithm='ball_tree', rtol=1e-4)
    kde.fit(data)
    Z = kde.score_samples(grid_ravel.T)
    Z = Z.reshape(grid_mesh[0].shape)
    time2 = time()
    t4.append(time2 - time1)
    print("time = ", (time2-time1))
    ########################################################################################################################
    # print("\nKDEpy: FFTKDE")
    # time1 = time()
    # kde = gaussian_kde(data.T, bw_method='scott')
    # kde = FFTKDE(kernel='gaussian', bw=0.2)
    # kde.fit(data)
    # grid, points =kde.evaluate(21)
    # print(grid.shape, points.shape)
    # z = points.reshape(21, 21, 21, 21).T
    # time2 = time()
    # t4.append(time2 - time1)
    # print("time = ", (time2-time1))

np.savetxt(os.path.join(path['plots'], 't1'), t1)
np.savetxt(os.path.join(path['plots'], 't2'), t2)
np.savetxt(os.path.join(path['plots'], 't3'), t3)
np.savetxt(os.path.join(path['plots'], 't4'), t4)
fig = plt.figure()
ax = plt.gca()
ax.semilogx(N_array, t1, label='scipy')
ax.semilogx(N_array, t2, label='scilearn')
ax.semilogx(N_array, t3, label='scilearn:kd_tree')
ax.semilogx(N_array, t4, label='scilearn:ball_tree')

ax.set_xlabel('N points')
ax.set_ylabel('time')
plt.legend()
fig.savefig(os.path.join(path['plots'], 'time'))
plt.close('all')