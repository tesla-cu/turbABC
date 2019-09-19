import os
import numpy as np
from time import time
import pyabc.kde as my_kde
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import matplotlib.pyplot as plt
from KDEpy import FFTKDE

path_base = '../runs_abc/'
path = {'output': os.path.join(path_base, 'output'), 'plots': os.path.join(path_base, 'plots_kde')}
if not os.path.isdir(path['plots']):
    os.makedirs(path['plots'])


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

########################################################################################################################


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

########################################################################################################################
#
########################################################################################################################
def kde_scipy(data, grid_ravel, bandwidth, **kwargs):
    kde = gaussian_kde(data.T, bw_method='scott', **kwargs)
    return grid_ravel, kde.evaluate(grid_ravel)


def kde_statsmodels_m(data, grid_ravel, bandwidth, **kwargs):
    kde = KDEMultivariate(data, bw='normal_reference', var_type='c'*len(data[0]), **kwargs)
    return grid_ravel, kde.pdf(grid_ravel)


def kde_sklearn(data, grid_ravel, bandwidth, **kwargs):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', **kwargs)
    kde.fit(data)
    log_pdf = kde.score_samples(grid_ravel.T)
    return grid_ravel, np.exp(log_pdf)


def kde_sklearn_kd_tree(data, grid_ravel, bandwidth, **kwargs):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='kd_tree', rtol=1e-4, **kwargs)
    kde.fit(data)
    log_pdf = kde.score_samples(grid_ravel.T)
    return grid_ravel, np.exp(log_pdf)


def kde_fftkde_kdepy(data, grid_ravel, bandwidth):
    kde = FFTKDE(kernel='gaussian', bw=bandwidth)
    kde.fit(data)
    x, y = kde.evaluate()
    return x.reshape((1, -1)), y

kde_funcs = [kde_statsmodels_m, kde_scipy, kde_sklearn, kde_sklearn_kd_tree, kde_fftkde_kdepy ]
kde_funcnames = ['Statsmodels-M', 'Scipy', 'Sklearn', 'Sklearn: KD tree', 'KDEpy: FFTKDE']


# from scipy.stats.distributions import norm
#
# # The grid we'll use for plotting
# x_grid = np.linspace(-4.5, 3.5, 32).reshape((1, -1))
# print('grid shape', x_grid.shape)
# # Draw points from a bimodal distribution in 1D
# np.random.seed(0)
# x = np.concatenate([norm(-1, 1.).rvs(400), norm(1, 0.3).rvs(100)]).reshape((-1, 1))
# print('data shape = ', x.shape)
# pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))
# bw = my_kde.bw_from_kdescipy(x)[0]
# print('bw = ', bw)
# #####################################################################
# fig = plt.figure()
# ax = plt.gca()
# for i in range(len(kde_funcnames)):
#     print(kde_funcnames[i])
#     grid, pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
#     print(grid.shape, pdf.shape)
#     ax.plot(grid[0], pdf.T, alpha=0.5, lw=3, label=kde_funcnames[i])
# ax.fill(x_grid[0], pdf_true[0], ec='gray', fc='gray', alpha=0.4)
# plt.legend()
# ax.set_xlim(-4.5, 3.5)
# fig.savefig(os.path.join(path['plots'], '1d_kde'))
# #####################################################################
# fig = plt.figure()
# ax = plt.gca()
# for bw_i in [0.1, bw, 0.7]:
#     grid, pdf = kde_sklearn(x, x_grid, bandwidth=bw_i)
#     ax.plot(grid[0], pdf.T, alpha=0.9, lw=3, label='bw = {}'.format(np.round(bw_i, 3)))
# ax.fill(x_grid[0], pdf_true[0], ec='red', lw=3, fc='gray', alpha=0.7)
# ax.hist(x, 32, fc='gray', histtype='stepfilled', alpha=0.7, normed=True)
#
# plt.legend()
# ax.set_xlim(-4.5, 3.5)
# fig.savefig(os.path.join(path['plots'], '1d_kde_kernels'))

C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
a, b = C_limits[:, 0], C_limits[:, 1]
num_bin_joint = 20
params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']

data_load = np.load(os.path.join(path['output'], 'tmp30.npz'))['accepted']
print(len(data_load))
N_array = [int(i*10**j) for j in range(2, 6) for i in [2.5, 5, 7.5, 10]]
N_array.append(int(2.5*1e6))
N_array.append(len(data_load))
print(N_array)

######################
# 1D compare methods
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
# for i in range(len(kde_funcnames)):
#     print(kde_funcnames[i])
#     grid, pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
#     ax.plot(grid[0], pdf.T, alpha=0.5, lw=3, label=kde_funcnames[i])
# ax.hist(x, 20, fc='gray', histtype='stepfilled', alpha=0.7, normed=True)
# plt.legend()
# ax.set_xlim(-1, 1)
# fig.savefig(os.path.join(path['plots'], '1d_kde_realdata'))
######################
# 1D compare speed
# t = np.empty((len(kde_funcs), len(N_array)))
# x_grid = np.linspace(-1, 1, 20).reshape((1, -1))
# for n, N in enumerate(N_array):
#     print('\nN = ', N)
#     ind = np.random.choice(np.arange(len(data_load)), size=N, replace=False)
#     data = 2 * (data_load[ind] - a) / (b - a) - 1
#     x = data[:, 0].reshape((-1, 1))
#     bw = my_kde.bw_from_kdescipy(x)[0]
#     print(bw)
#     for i in range(len(kde_funcnames)):
#         print(kde_funcnames[i], end='\t')
#         t0 = time()
#         pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
#         t1 = time()
#         t[i, n] = (t1-t0)
#         print(t[i, n])
#     fig = plt.figure()
#     ax = plt.gca()
#     for i in range(len(kde_funcnames)):
#         ax.loglog(N_array, t[i], alpha=0.5, lw=3, label=kde_funcnames[i])
#     plt.legend()
#     fig.savefig(os.path.join(path['plots'], '1d_time_realdata'))
# exit()
#######################################################################################################################
# 4D compare speed
t = np.empty((len(kde_funcs), len(N_array)))
_, x_grid = my_kde.grid_for_kde(np.zeros(4), np.ones(4), 20)
print(x_grid.shape)
for n, N in enumerate(N_array[:10]):
    print('\nN = ', N)
    ind = np.random.choice(np.arange(len(data_load)), size=N, replace=False)
    x = 2 * (data_load[ind] - a) / (b - a) - 1
    # x = data
    bw = my_kde.bw_from_kdescipy(x)[0]
    print(bw)
    for i in range(len(kde_funcnames)):
        # if not (kde_funcnames[i] == 'Statsmodels-M' and N > 25000):
        print(kde_funcnames[i], end='\t')
        t0 = time()
        grid, pdf = kde_funcs[i](x, x_grid, bandwidth=bw)
        t1 = time()
        t[i, n] = (t1-t0)
        print(t[i, n])
    fig = plt.figure()
    ax = plt.gca()
    for i in range(len(kde_funcnames)):
        # if kde_funcnames[i] == 'Statsmodels-M':
        #     ax.loglog(N_array[:9], t[i, :9], alpha=0.5, lw=3, label=kde_funcnames[i])
        ax.loglog(N_array[:10], t[i, :10], alpha=0.5, lw=3, label=kde_funcnames[i])
    plt.legend()
    fig.savefig(os.path.join(path['plots'], '4d_time_realdata'))
    np.savez(os.path.join(path['plots'], '4d_time.npz'), t=t)

# t = [[31.663392782211304, 37.94662022590637, 44.902695417404175, 52.76849985122681, 87.86744284629822, 148.54778671264648, 208.50601196289062],
#      [1.9641737937927246, 3.4374029636383057, 5.148638010025024, 6.937317132949829, 17.04867649078369, 34.272719383239746, 51.42552208900452, 67.9982602596283, 173.63636374473572, 346.3071653842926],
#      [4.159970283508301, 8.278577327728271, 14.804991960525513, 16.67172360420227, 39.6780047416687, 83.30278706550598, 124.94936847686768, 151.51159405708313, 389.64109086990356, 729.719051361084],
#      [4.537948369979858, 8.017235040664673, 12.350555419921875, 16.143917560577393, 29.088284969329834, 49.74363374710083, 69.50050139427185, 78.39611744880676, 154.18389296531677, 244.8754117488861]]
# fig = plt.figure()
# ax = plt.gca()
# for i in range(4):
#     ax.loglog(N_array[:len(t[i])], t[i], alpha=0.5, lw=3, label=kde_funcnames[i])
# plt.legend()
# fig.savefig(os.path.join(path['plots'], '4d_time_realdata'))

