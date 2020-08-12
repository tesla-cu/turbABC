import os
import string
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors
plt.style.use('dark_background')

mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 1 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
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


def plot_marginal_change(data_folders, params_names, C_limits, n_bins, plot_folder, nominal_values=None, smooth=''):
    N_params = len(C_limits)
    print('N_params', N_params)
    colormap = plt.cm.gist_ncar
    color_array = [colormap(i) for i in np.linspace(0, 1, len(data_folders))]
    plt.gca().set_prop_cycle(color=color_array)
    labels = []
    if N_params == 4:
        fig_width, fig_height = fig_size(oneandhalf_column)
        fig, axarr = plt.subplots(nrows=1, ncols=N_params, figsize=(fig_width, 0.8 * fig_height))
    if N_params == 5:
        fig_width, fig_height = fig_size(double_column)
        fig, axarr = plt.subplots(nrows=1, ncols=N_params, figsize=(fig_width, 0.6 * fig_height))
    axarr[0].yaxis.set_major_formatter(plt.NullFormatter())
    axarr[0].yaxis.set_major_locator(plt.NullLocator())
    for folder in data_folders:
        x = os.path.basename(os.path.normpath(folder))[2:]
        # if smooth:
        #     MAP_x = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(n_bins)))
        #     MAP_x = MAP_x.reshape((-1, N_params))
        labels.append('x = {}\%'.format(x))
        for i in range(N_params):
            data_marg = np.loadtxt(os.path.join(folder, f'marginal_{smooth}{i}'))
            # if smooth:
            #     for map in MAP_x:
            #         MAP_y = np.interp(map[i], data_marg[0], data_marg[1])
            #         axarr[i].scatter(map[i], MAP_y, color='r', s=10, zorder=2)
            axarr[i].plot(data_marg[0], data_marg[1], zorder=1)
            axarr[i].set_xlabel(params_names[i])
            axarr[i].set_xlim(C_limits[i])
            axarr[i].axis(y_min=0)
            # axarr[i].ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
    # custom_lines = [Line2D([0], [0], color='b', ls='--', lw=4)]
    # legend_labels = ['nominal value']
    # for i, label in enumerate(labels):
    #     custom_lines.append(Line2D([0], [0], color=color_array[i], lw=1))
    #     legend_labels.append(label)
    plt.legend(labels, ncol=3, loc='upper center',
               bbox_to_anchor=[-1.0, 1.35], labelspacing=0.0,
               handletextpad=0.5, handlelength=1.5,
               fancybox=False, shadow=False, frameon=False)
    if nominal_values:
        for i in range(N_params):
            axarr[i].axvline(nominal_values[i], color='b', linestyle='--')
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.06, hspace=0.1, bottom=0.18, top=0.82)
    fig.savefig(os.path.join(plot_folder, f'marginal_change_{smooth}'))
    plt.close('all')


def plot_MAP_confidence_change(data_folders, params_names, num_bin_kde, C_limits, plot_folder):
    N_params = len(params_names)
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 1, len(data_folders))])
    fig_width, fig_height = fig_size(oneandhalf_column)
    fig, axarr = plt.subplots(nrows=1, ncols=N_params, sharey=True, figsize=(fig_width, 0.7 * fig_height))
    MAP = []
    # np.empty((len(data_folders), N_params))
    conf_level = [0.05, 0.1, 0.25]
    confidence = np.empty((len(data_folders), N_params, len(conf_level), 2))
    quantile = np.empty_like(confidence)

    x = np.empty(len(data_folders))
    for i, folder in enumerate(data_folders):
        x[i] = float(os.path.basename(os.path.normpath(folder))[2:])
        print('x = ', x[i])
        MAP.append(np.loadtxt(os.path.join(folder, 'C_final_smooth'.format(num_bin_kde))))
        for j, level in enumerate(conf_level):
            confidence[i, :, j] = np.loadtxt(os.path.join(folder, 'confidence_{}'.format(int(100 * (1 - level)))))
            quantile[i, :, j] = np.loadtxt(os.path.join(folder, 'quantile_{}'.format(int(100 * (1 - level)))))
    print(MAP)
    ind = np.argsort(x)
    x = x[ind]
    MAP = np.array(MAP)[ind]
    print(MAP)
    confidence = confidence[ind]
    quantile = quantile[ind]
    colors = ['b', 'g', 'y']
    colors2 = ['orange', 'k', 'magenta']
    for i in range(N_params):
        for j in range(len(x)):
            MAP[j] = MAP[j].reshape((-1, N_params))
            axarr[i].scatter(MAP[j][i], x[j], s=10, color='r', zorder=2)
        for j in range(len(conf_level)):
            axarr[i].semilogy(confidence[:, i, j, 0], x, color=colors[j])
            axarr[i].semilogy(confidence[:, i, j, 1], x, color=colors[j])
            axarr[i].semilogy(quantile[:, i, j, 0], x, color=colors2[j])
            axarr[i].semilogy(quantile[:, i, j, 1], x, color=colors2[j])
        axarr[i].set_xlabel(params_names[i])
        axarr[i].set_xlim(C_limits[i])

    axarr[0].set_ylabel('x [\%]')
    fig.subplots_adjust(left=0.12, right=0.98, wspace=0.1, hspace=0.1, bottom=0.2, top=0.8)

    custom_lines = [Line2D([0], [0], marker='o', color='r', lw=1)]
    legend_text = ['MAP']
    for j, level in enumerate(conf_level):
        custom_lines.append(Line2D([0], [0], color=colors[j], linestyle='-', lw=1))
        legend_text.append('{}\% confidence'.format(int(100 * (1 - level))))
    axarr[1].legend(custom_lines, legend_text, loc='upper center',
                    bbox_to_anchor=[0.7, 1.35], frameon=False,
                    labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
                    fancybox=False, shadow=False, ncol=2)

    fig.savefig(os.path.join(plot_folder, 'MAP_change'))
    plt.close('all')


def plot_eps_change(data_folders, plot_folder):
    fig_width, fig_height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(0.5 * fig_width, 0.5 * fig_height))
    ax = plt.gca()
    x = np.empty(len(data_folders))
    eps = np.empty_like(x)
    for i, folder in enumerate(data_folders):
        x[i] = float(os.path.basename(os.path.normpath(folder))[2:])
        eps[i] = np.loadtxt(os.path.join(folder, 'eps'))
    ind = np.argsort(x)
    x = x[ind]
    eps = eps[ind]
    ax.plot(eps, x, '-o')
    ax.set_ylabel('x [\%]')
    ax.set_xlabel('epsilon')
    fig.subplots_adjust(left=0.2, right=0.98, wspace=0.05, hspace=0.1, bottom=0.25, top=0.95)

    fig.savefig(os.path.join(plot_folder, 'eps_change'))
    plt.close('all')