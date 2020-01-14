import os
import glob
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from overflow.sumstat import TruthData, GridData
plt.style.use('dark_background')

mpl.rcParams['font.size'] = 12
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


basefolder = '../'

output_folder = os.path.join(basefolder, 'output_map', )
plot_folder = os.path.join(basefolder, 'plots', )
exp_folder = os.path.join(basefolder, 'overflow/valid_data', )
nominal_folder = os.path.join(basefolder, 'nominal', 'nominal_data', )

if not os.path.isdir(plot_folder):
    os.makedirs(plot_folder)
params_names = [r'$\beta^*$', r'$\sigma_{w1}$', r'$\beta_1$', r'$\beta_2$']

Truth = TruthData(exp_folder, ['cp', 'u', 'uv'])
sumstat_true = Truth.sumstat_true
Grid = GridData(exp_folder)

cp_nominal = np.fromfile(os.path.join(nominal_folder, 'cp_all.bin'), dtype=float)
u_nominal = np.fromfile(os.path.join(nominal_folder, 'u_slice.bin'), dtype=float)
uv_nominal = np.fromfile(os.path.join(nominal_folder, 'uv_slice.bin'), dtype=float)

labels = [r'trained on $C_p$', r"trained on $\overline{u}$", r'trained on $\overline{u^{\prime}v^{\prime}}$', 'trained all']
x_list = ['30', '10', '5', '3', '1', '05']


def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height


def plot_cp(map_cp, ind):
    experiment_cp = Truth.cp
    x_min, x_max = np.min(experiment_cp[:, 0]), np.max(experiment_cp[:, 0])
    # ind = np.where(np.logical_and(g.Grid.grid_x > x_min, g.Grid.grid_x < x_max))[0]
    # Cp data from overflow

    width, height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(width, 1.25 * height))
    ax = plt.gca()
    ax.invert_yaxis()

    for i, map in enumerate(map_cp):
        if i == 3:
            ax.plot(Grid.grid_x, map, label=labels[i])
    ax.plot(Grid.grid_x, cp_nominal, label='nominal')
    ax.plot(experiment_cp[:, 0], experiment_cp[:, 1], 'r+', linewidth=4, label='experiment')
    ax.grid(True)
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$C_p$')
    ax.axis(xmin=0.4, xmax=1.8, ymin=0.2, ymax=-0.9)
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.3, right=0.97, bottom=0.2, top=0.9)  # presentation

    fig.savefig(os.path.join(plot_folder, 'cp_map{}percent'.format(ind)))
    plt.close('all')


def plot_u(map_u, slice, ind):

    y = np.linspace(0, 0.14, 100)
    experiment_u = Truth.u[slice]
    u_nom = u_nominal[slice*100:(slice+1)*100]

    width, height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(width, 1.25 * height))
    ax = plt.gca()
    for i, map in enumerate(map_u):
        if i == 3:
            ax.plot(map, y, label=labels[i])
    ax.plot(u_nom, y, label='nominal')
    ax.plot(experiment_u[:, 0], experiment_u[:, 1], 'r+', linewidth=4, label='experiment')
    ax.grid(True)
    ax.set_xlabel(r'$\overline{u}/U_{\infty}$')
    ax.set_ylabel(r'$y/c$')
    # ax.axis(ymin=0, ymax=0.14, xmin=-0.25, xmax=1.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.3, right=0.97, bottom=0.2, top=0.9)  # presentation

    fig.savefig(os.path.join(plot_folder, 'u_map{}percent_{}slice'.format(ind, slice)))
    plt.close('all')


def plot_uv(map_u, slice, ind):
    y = np.linspace(0, 0.14, 100)
    experiment_uv = Truth.uv[slice]
    uv_nom = uv_nominal[slice * 100:(slice + 1) * 100]

    width, height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(width, 1.25 * height))
    ax = plt.gca()
    for i, map in enumerate(map_u):
        if i == 3:
            ax.plot(-map, y, label=labels[i])
    ax.plot(-uv_nom, y, label='nominal')
    ax.plot(experiment_uv[:, 0], experiment_uv[:, 1], 'r+', linewidth=4, label='experiment')
    ax.grid(True)
    ax.set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\infty}$')
    ax.set_ylabel(r'$y/c$')
    # ax.axis(ymin=0, ymax=0.14, xmin=-0.02, xmax=0)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.3, right=0.97, bottom=0.2, top=0.9)  # presentation

    fig.savefig(os.path.join(plot_folder, 'uv_map{}percent_{}slice'.format(ind, slice)))
    plt.close('all')


sum_stat = ['cp', 'u', 'uv', 'all']
folders = [os.path.join(output_folder, "calibration_map_{}".format(i)) for i in sum_stat ]

cp_maps = np.empty((0, len(cp_nominal)))
u_maps = np.empty((0, len(u_nominal)))
uv_maps = np.empty((0, len(u_nominal)))
for i, folder in enumerate(folders):
    cp_maps = np.vstack((cp_maps, np.fromfile(os.path.join(folder, 'cp_all.bin')).reshape(-1, 721)))
    u_maps = np.vstack((u_maps, np.fromfile(os.path.join(folder, 'u_slice.bin')).reshape(-1, 800)))
    uv_maps = np.vstack((uv_maps, np.fromfile(os.path.join(folder, 'uv_slice.bin')).reshape(-1, 800)))




for j, x in enumerate(x_list):
    p = np.array([j+i*6 for i in range(len(sum_stat))])
    plot_cp(cp_maps[p], x_list[j])
    for slice in range(8):
        # print(x_list[j], slice, '\t', uv_maps[p, slice * 100:(slice + 1) * 100][0])
        plot_u(u_maps[p, slice*100:(slice+1)*100], slice, x_list[j])
        plot_uv(uv_maps[p, slice * 100:(slice + 1) * 100], slice, x_list[j])
