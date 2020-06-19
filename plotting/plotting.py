import os
import numpy as np
# from sklearn.metrics import r2_score
import postprocess.regression as regression
import matplotlib as mpl

mpl.use('pdf')
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from rans_ode.sumstat import TruthData

plt.style.use('dark_background')

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 1
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
# dessertation size
single_column = 235
oneandhalf_column = 352
double_column = 470


def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height


folder = './plots/'


def plot_1d_pdf_change(data_folders, params_names, C_limits, num_bin_kde, plot_folder):
    # colormap = plt.cm.gist_ncar

    labels = []
    fig_width, fig_height = fig_size(single_column)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    colors = ['k', 'g', 'b', 'y', 'm', 'orange']
    # ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(data_folders))])
    x = []
    for f, folder in enumerate(data_folders):
        x.append(float(os.path.basename(os.path.normpath(folder))[2:]))
    ind = np.argsort(x)
    for f, folder in enumerate(np.array(data_folders)[ind]):
        MAP_x = np.loadtxt(os.path.join(folder, 'C_final_smooth'))

        # labels.append(int(float(x)))
        pdf = np.load(os.path.join(folder, 'Z.npz'))['Z']
        pdf_x = np.load(os.path.join(folder, 'Z.npz'))['grid']
        MAP_y = np.interp(MAP_x, pdf_x, pdf)
        ax.scatter(MAP_x, MAP_y, color='r', s=10, zorder=2)
        ax.plot(pdf_x, pdf, color=colors[f], zorder=1, label='x={}\%'.format(x[ind[f]]))
    ax.set_xlabel(params_names[0])
    ax.set_xlim([1.2, 1.7])
    fig.subplots_adjust(left=0.15, right=0.92, bottom=0.2, top=0.92)
    plt.legend(loc=0)

    fig.savefig(os.path.join(plot_folder, 'marginal_change'))
    plt.close('all')


def plot_marginal_change_with_regression(data_folders, folder_reg, params_names, C_limits, num_bin_kde, plot_folder):
    N_params = len(params_names)
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 1, len(data_folders))])
    labels = []
    fig, axarr = plt.subplots(nrows=1, ncols=N_params, sharey=True, figsize=(fig_width, 0.8 * fig_height))
    for folder in data_folders:
        x = os.path.basename(os.path.normpath(folder))[2:]
        MAP_x = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
        MAP_x = MAP_x.reshape((-1, N_params))
        MAP_reg = np.loadtxt(os.path.join(folder_reg, 'C_final_smooth{}'.format(num_bin_kde)))
        labels.append('x = {}\%'.format(x))
        for i in range(N_params):
            data_marg = np.loadtxt(os.path.join(folder, 'marginal_smooth{}'.format(i)))
            for map in MAP_x:
                MAP_y = np.interp(map[i], data_marg[0], data_marg[1])
                axarr[i].scatter(map[i], MAP_y, color='r', s=10, zorder=3)
            axarr[i].plot(data_marg[0], data_marg[1], zorder=2)
            axarr[i].axvline(MAP_reg[i], color='b', linestyle='--', zorder=1)
            axarr[i].yaxis.set_major_formatter(plt.NullFormatter())
            axarr[i].set_xlabel(params_names[i])

            # axarr[i].set_xlim(C_limits[i])
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.05, hspace=0.1, bottom=0.2, top=0.8)

    plt.legend(labels, ncol=3, loc='upper center',
               bbox_to_anchor=[-0.6, 1.3], labelspacing=0.0,
               handletextpad=0.5, handlelength=1.5,
               fancybox=True, shadow=True)

    # custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
    #                 Line2D([0], [0], color=colors[1], linestyle='-', lw=1),
    #                 Line2D([0], [0], color=colors[2], linestyle='-', lw=1)]
    # axarr[0, 1].legend(custom_lines, ['true data', '3 parameters', '4 parameters'], loc='upper center',
    #                    bbox_to_anchor=(0.99, 1.35), frameon=False,
    #                    fancybox=False, shadow=False, ncol=3)

    fig.savefig(os.path.join(plot_folder, 'marginal_change_with_regression'))
    plt.close('all')








def plot_dist_pdf(path, dist, x):
    fig_width, fig_height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.hist(dist, bins=100, alpha=0.8)
    eps = np.percentile(dist, q=int(x * 100))
    print('eps =', eps)
    ax.axvline(eps)
    # ax.set_xlabel('d')
    # ax.set_ylabel('pdf(d)')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(path, 'dist'))
    plt.close('all')
    return eps


def plot_results(truth, results, path, name):
    fig_width, fig_height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.plot(truth, 'r')
    ax.plot(results)
    fig.savefig(os.path.join(path, name))
    plt.close('all')


def plot_bootstrapping_pdf(path, dist, x, i, C_limit, C_final):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.hist(dist, bins=20, alpha=0.8, range=C_limit)
    q1 = np.percentile(dist, q=x)
    ax.axvline(q1, color='g', label=r'$2.5\%$')
    q2 = np.percentile(dist, q=100 - x)
    ax.axvline(q2, color='g', label=r'$97.5\%$')
    ax.axvline(C_final, color='r', label=r'$C_{max}$')
    ax.set_xlabel(r'$C_{}$'.format(i))
    ax.set_ylabel(r'pdf')
    plt.legend()
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(path['plots'], 'bootstrapping' + str(i)))
    plt.close('all')
    return q1, q2


def plot_1d_dist_scatter(data, C_limits, params_name, x_list, eps_list, plot_folder):
    fig_width, fig_height = fig_size(single_column)
    fig = plt.figure(figsize=(fig_width, fig_width))
    ax = plt.axes()
    colors = ['r', 'g', 'k', 'y', 'm', 'orange']
    ax.axis(xmin=C_limits[0], xmax=C_limits[1], ymax=1.005 * np.max(eps_list), ymin=0.98 * np.min(eps_list))
    ax.scatter(data[:, 0], data[:, 1], marker=".", color='blue')
    for i, eps in enumerate(eps_list):
        ax.axhline(eps, color=colors[i], label='{}\%'.format(x_list[i][2:]))
    ax.set_xlabel(params_name)
    ax.set_ylabel('distance')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.legend()
    fig.subplots_adjust(left=0.245, right=0.96, bottom=0.21, top=0.97)
    fig.savefig(os.path.join(plot_folder, 'scatter_plot'))
    plt.close('all')


def plot_sampling_hist(samples, C_limits, params_name, plot_folder):
    fig_width, fig_height = fig_size(single_column)
    fig = plt.figure(figsize=(fig_width, fig_width))
    ax = plt.axes()
    ax.hist(samples, range=C_limits, alpha=0.6)
    ax.set_xlabel(params_name)
    ax.set_title('Prior')
    fig.subplots_adjust(left=0.245, right=0.96, bottom=0.21, top=0.97)
    fig.savefig(os.path.join(plot_folder, 'prior'))
    plt.close('all')


def plot_regression(c, sum_stat, dist, solution, params_name, plot_folder):
    def line(i, x):
        return solution[i, 0] + solution[i, 1] * x

    true = TruthData(valid_folder='../rans_ode/valid_data/', case='impulsive')
    full_true = true.sumstat_true
    for i in range(len(full_true)):
        new = sum_stat[:, i] - full_true[i]
        x = np.linspace(np.min(new), np.max(new), 100)
        print(solution[i])
        print(full_true[4])
        y = line(i, x)
        R = regression.calc_r2_score(c[:, 0], line(i, new))
        # R1 = r2_score(c[:, 0], line(i, new))
        # print('R', i, R, R1)
        fig_width, fig_height = fig_size(single_column)
        fig = plt.figure(figsize=(fig_width, 1.1 * fig_height))
        ax = plt.axes()
        ax.scatter(new, c, marker=".")
        ax.axvline(0)
        ax.plot(x, y, color='w')
        ax.set_ylabel(params_name)
        ax.set_xlabel(r'$\mathcal{S} - \mathcal{S}_{true}$')
        ax.set_title(r'Linear regression, $R^2 = {}$'.format(np.round(R, 3)))
        fig.subplots_adjust(left=0.245, right=0.96, bottom=0.21, top=0.88)
        fig.savefig(os.path.join(plot_folder, 'regression_full{}'.format(i)))

        # plot residuals
        fig = plt.figure(figsize=(0.75 * fig_width, 0.5 * fig_width))
        ax = plt.axes()
        ax.scatter(new, (c[:, 0] - line(i, new)) / (np.max(c) - np.min(c)), marker=".")
        ax.axhline(0.0, color='w')
        ax.set_ylabel(params_name)
        ax.set_xlabel(r'$\mathcal{S} - \mathcal{S}_{true}$')
        ax.set_title('Linear regression')
        fig.subplots_adjust(left=0.245, right=0.96, bottom=0.21, top=0.9)
        fig.savefig(os.path.join(plot_folder, 'regression_res{}'.format(i)))

    plt.close('all')
    # y = -1.55951292*x+1.59051902
    # fig = plt.figure(figsize=(0.75 * fig_width, 0.5 * fig_width))
    # ax = plt.axes()
    # ax.scatter(dist, c, marker=".", color='blue')
    # ax.axvline(0.0)
    # ax.plot(y, x, color='k')
    # ax.set_ylabel(params_name)
    # ax.set_xlabel('distance')
    # ax.set_title('regression distance')
    # fig.subplots_adjust(left=0.245, right=0.96, bottom=0.21, top=0.9)
    # fig.savefig(os.path.join(plot_folder, 'regression_dist'))
    # plt.close('all')
