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

output_folder = os.path.join(basefolder, 'overflow_results/output', )
plot_folder = os.path.join(basefolder, 'overflow_results/plots_map_chains', )
exp_folder = os.path.join('../overflow/valid_data', )
nominal_folder = os.path.join('../nominal/nominal_data/', )

if not os.path.isdir(plot_folder):
    os.makedirs(plot_folder)

# params_names = [r'$\beta^*$', r'$\sigma_{w1}$', r'$\beta_1$', r'$\beta_2$']

Truth = TruthData(exp_folder, ['cp', 'u', 'uv', 'x_separation'])
sumstat_true = Truth.sumstat_true
Grid = GridData(exp_folder)

cp_nominal = np.fromfile(os.path.join(nominal_folder, 'cp_all.bin'), dtype=float)
u_nominal = np.fromfile(os.path.join(nominal_folder, 'u_slice.bin'), dtype=float)
uv_nominal = np.fromfile(os.path.join(nominal_folder, 'uv_slice.bin'), dtype=float)

labels = [r'trained on $C_p$', r"trained on $\overline{u}$", r'trained on $\overline{u^{\prime}v^{\prime}}$',
          r'trained on $C_p$ and $\overline{u}$', r'trained on $C_p$, $\overline{u}$ and $\overline{u^{\prime}v^\prime}$',
          r'trained on $C_p$, $\overline{u}$ and $\overline{u^{\prime}v^\prime}$'+'\n'+r'if $err_{sep}< 0.25$']
x_list = ['30', '20', '10', '5', '3']
profiles_labels = [r'$x/c = -0.25$', r'$x/c = 0.688$', r'$x/c = 0.813$', r'$x/c = 0.938$',
                   r'$x/c = 1.0$', r'$x/c = 1.125$', r'$x/c = 1.25$', r'$x/c = 1.375$']
colors_arr = plt.cm.get_cmap('cool')(np.linspace(0, 1, len(labels)))
for x in x_list:
    newdir = os.path.join(plot_folder, f'percent{x}',)
    if not os.path.isdir(newdir):
        os.mkdir(newdir)


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

    width, height = fig_size(double_column)
    fig = plt.figure(figsize=(width, 0.8*height))
    ax = plt.gca()
    ax.invert_yaxis()

    for i, map in enumerate(map_cp):
        ax.plot(Grid.grid_x, map, label=labels[i], color=colors_arr[i])
    ax.plot(Grid.grid_x, cp_nominal, label='nominal', color='r')
    ax.plot(experiment_cp[:, 0], experiment_cp[:, 1], 'w+', linewidth=6, label='experiment')
    ax.grid(True)
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$C_p$')
    ax.axis(xmin=0.4, xmax=1.8, ymin=0.2, ymax=-0.9)
    ax.legend(bbox_to_anchor=[-1., 1], loc='upper left',
              frameon=False,
              labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
              fancybox=False, shadow=False,)
    fig.subplots_adjust(left=0.5, right=0.98, bottom=0.18, top=0.97)  # presentation

    fig.savefig(os.path.join(plot_folder, f'percent{ind}', f'cp_map'))
    plt.close('all')


def plot_cp_percent(map_cp, stat):
    experiment_cp = Truth.cp
    x_min, x_max = np.min(experiment_cp[:, 0]), np.max(experiment_cp[:, 0])
    # ind = np.where(np.logical_and(g.Grid.grid_x > x_min, g.Grid.grid_x < x_max))[0]
    # Cp data from overflow

    width, height = fig_size(double_column)
    fig = plt.figure(figsize=(width, 0.8*height))
    ax = plt.gca()
    ax.invert_yaxis()

    for i, map in enumerate(map_cp):
        ax.plot(Grid.grid_x, map, label=f'x = {x_list[i]}%')
    ax.plot(Grid.grid_x, cp_nominal, label='nominal')
    ax.plot(experiment_cp[:, 0], experiment_cp[:, 1], 'w+', linewidth=4, label='experiment')
    ax.grid(True)
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$C_p$')
    ax.axis(xmin=0.4, xmax=1.8, ymin=0.2, ymax=-0.9)
    ax.legend(bbox_to_anchor=[-1., 1], loc='upper left',
              frameon=False,
              labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
              fancybox=False, shadow=False,)
    fig.subplots_adjust(left=0.5, right=0.98, bottom=0.18, top=0.97)  # presentation

    fig.savefig(os.path.join(plot_folder, f'cp_map_{stat}'))
    plt.close('all')


def plot_u_subplot(map_u, ind):

    y = np.linspace(0, 0.14, 100)

    width, height = fig_size(double_column)
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(width, height))
    for slice in range(8):
        i, j = np.divmod(slice, 4, dtype=int)
        experiment_u = Truth.u[slice]
        u_nom = u_nominal[slice * 100:(slice + 1) * 100]
        map_u_slice = map_u[:, slice*100:(slice+1)*100]
        for m, map in enumerate(map_u_slice):
            ax[i, j].plot(map[1:], y[1:], label=labels[m])
        ax[i, j].plot(u_nom, y, label='nominal')
        ax[i, j].plot(experiment_u[:, 0], experiment_u[:, 1], 'r+', linewidth=4, label='experiment')
        ax[i, j].grid(True)
        ax[i, j].tick_params( which='major', pad=1.)
        ax[-1, j].set_xlabel(r'$\overline{u}/U_{\infty}$', labelpad=0.0)
        ax[i, 0].set_ylabel(r'$y/c$', labelpad=0.0)
        ax[i, j].axis(ymin=0, ymax=0.13, xmin=-0.25, xmax=1.4)
    ax[0, 0].legend(bbox_to_anchor=[2., 1.55], loc='upper center', frameon=False,
                    labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
                    fancybox=False, shadow=False, ncol=3)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.83, hspace=0.05, wspace=0.05)  # presentation

    fig.savefig(os.path.join(plot_folder, f'percent{ind}', f'u_map'))
    plt.close('all')


def plot_u_subplot_percent(map_u, stat):

    y = np.linspace(0, 0.14, 100)
    width, height = fig_size(double_column)
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(width, height))
    for slice in range(8):
        i, j = np.divmod(slice, 4, dtype=int)
        experiment_u = Truth.u[slice]
        u_nom = u_nominal[slice * 100:(slice + 1) * 100]
        map_u_slice = map_u[:, slice*100:(slice+1)*100]
        for m, map in enumerate(map_u_slice):
            ax[i, j].plot(map[1:], y[1:], label=f'x = {x_list[m]}%')
        ax[i, j].plot(u_nom, y, label='nominal')
        ax[i, j].plot(experiment_u[:, 0], experiment_u[:, 1], 'r+', linewidth=4, label='experiment')
        ax[i, j].grid(True)
        ax[i, j].set_title(profiles_labels[slice], pad=0.05)
        ax[i, j].tick_params( which='major', pad=1.)
        ax[-1, j].set_xlabel(r'$\overline{u}/U_{\infty}$', labelpad=0.0)
        ax[i, 0].set_ylabel(r'$y/c$', labelpad=0.0)
        ax[i, j].axis(ymin=0, ymax=0.13, xmin=-0.25, xmax=1.4)
    ax[0, 0].legend(bbox_to_anchor=[2., 1.45], loc='upper center', frameon=False,
                    labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
                    fancybox=False, shadow=False, ncol=3)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.87, hspace=0.15, wspace=0.05)  # presentation

    fig.savefig(os.path.join(plot_folder, f'u_map_{stat}'))
    plt.close('all')


def plot_u(map_u, slice, ind):

    y = np.linspace(0, 0.14, 100)
    experiment_u = Truth.u[slice]
    u_nom = u_nominal[slice * 100:(slice + 1) * 100]

    width, height = fig_size(double_column)
    fig = plt.figure(figsize=(width, 0.8*height))
    ax = plt.gca()
    for i, map in enumerate(map_u):
        ax.plot(map, y, label=labels[i], color=colors_arr[i])
    ax.plot(u_nom, y, label='nominal', color='r')
    ax.plot(experiment_u[:, 0], experiment_u[:, 1], 'w+', linewidth=4, label='experiment')
    ax.grid(True)

    ax.set_xlabel(r'$\overline{u}/U_{\infty}$', labelpad=0.0)
    ax.set_ylabel(r'$y/c$', labelpad=0.0)
    ax.axis(ymin=0, ymax=0.13, xmin=-0.25, xmax=1.4)
    ax.legend(bbox_to_anchor=[-0.8, 1], loc='upper left',
              frameon=False,
              labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
              fancybox=False, shadow=False, )
    fig.subplots_adjust(left=0.44, right=0.98, bottom=0.15, top=0.97)  # presentation
    fig.savefig(os.path.join(plot_folder, f'percent{ind}', f'u_map_{slice}slice'))
    plt.close('all')


def plot_uv_subplot(map_uv, ind):

    y = np.linspace(0, 0.14, 100)

    width, height = fig_size(double_column)
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(width, height))
    for slice in range(8):
        i, j = np.divmod(slice, 4, dtype=int)
        experiment_uv = Truth.uv[slice]
        uv_nom = uv_nominal[slice * 100:(slice + 1) * 100]
        map_uv_slice = map_uv[:, slice*100:(slice+1)*100]
        for m, map in enumerate(map_uv_slice):
            ax[i, j].plot(-map[1:], y[1:], label=labels[m])
        ax[i, j].plot(-uv_nom, y, label='nominal')
        ax[i, j].plot(experiment_uv[:, 0], experiment_uv[:, 1], 'r+', linewidth=4, label='experiment')
        ax[i, j].grid(True)
        ax[i, j].tick_params(which='major', pad=1.)
        ax[-1, j].set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\infty}$', labelpad=0.5)
        ax[i, 0].set_ylabel(r'$y/c$', labelpad=0.0)
        ax[i, j].axis(ymin=0, ymax=0.13, xmin=-0.02, xmax=0.001)
    ax[0, 0].legend(bbox_to_anchor=[2., 1.55], loc='upper center', frameon=False,
                    labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
                    fancybox=False, shadow=False, ncol=3)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.83, hspace=0.05, wspace=0.05)  # presentation

    fig.savefig(os.path.join(plot_folder, f'percent{ind}', f'uv_map'))
    plt.close('all')


def plot_uv_subplot_percent(map_uv, stat):

    y = np.linspace(0, 0.14, 100)

    width, height = fig_size(double_column)
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(width, height))
    for slice in range(8):
        i, j = np.divmod(slice, 4, dtype=int)
        experiment_uv = Truth.uv[slice]
        uv_nom = uv_nominal[slice * 100:(slice + 1) * 100]
        map_uv_slice = map_uv[:, slice*100:(slice+1)*100]
        for m, map in enumerate(map_uv_slice):
            ax[i, j].plot(-map[1:], y[1:], label=f'x = {x_list[m]}%')
        ax[i, j].plot(-uv_nom, y, label='nominal')
        ax[i, j].plot(experiment_uv[:, 0], experiment_uv[:, 1], 'w+', linewidth=4, label='experiment')
        ax[i, j].grid(True)
        ax[i, j].tick_params(which='major', pad=1.)
        ax[-1, j].set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\infty}$', labelpad=0.5)
        ax[i, 0].set_ylabel(r'$y/c$', labelpad=0.0)
        ax[i, j].axis(ymin=0, ymax=0.13, xmin=-0.02, xmax=0.001)
    ax[0, 0].legend(bbox_to_anchor=[2., 1.55], loc='upper center', frameon=False,
                    labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
                    fancybox=False, shadow=False, ncol=3)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.83, hspace=0.05, wspace=0.05)  # presentation

    fig.savefig(os.path.join(plot_folder, f'uv_map_{stat}'))
    plt.close('all')


def plot_uv(map_u, slice, ind):
    y = np.linspace(0, 0.14, 100)
    experiment_uv = Truth.uv[slice]
    uv_nom = uv_nominal[slice * 100:(slice + 1) * 100]

    width, height = fig_size(double_column)
    fig = plt.figure(figsize=(width, 0.8 * height))
    ax = plt.gca()
    for i, map in enumerate(map_u):
        ax.plot(-map, y, label=labels[i], color=colors_arr[i])
    ax.plot(-uv_nom, y, label='nominal', color='r')
    ax.plot(experiment_uv[:, 0], experiment_uv[:, 1], 'w+', linewidth=4, label='experiment')
    ax.grid(True)
    ax.set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\infty}$')
    ax.set_ylabel(r'$y/c$')
    # ax.axis(ymin=0, ymax=0.13, xmin=-0.25, xmax=1.4)
    ax.legend(bbox_to_anchor=[-0.8, 1], loc='upper left',
              frameon=False,
              labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
              fancybox=False, shadow=False, )
    fig.subplots_adjust(left=0.44, right=0.98, bottom=0.15, top=0.97)  # presentation

    fig.savefig(os.path.join(plot_folder, f'percent{ind}', f'uv_map_{slice}slice'))
    plt.close('all')


def plot_separation(x_maps):

    experiment_x = np.array([0.7, 1.1])
    x_normal = np.array([0.64, 1.18])

    y = np.arange(len(x_list))
    width, height = fig_size(double_column)
    fig = plt.figure(figsize=(width, 0.8 * height))
    ax = plt.gca()
    for map_i in range(len(labels)):
        ax.scatter(x_maps[map_i*len(x_list):(map_i+1)*len(x_list), 0], y, label=labels[map_i])
        ax.scatter(x_maps[map_i * len(x_list):(map_i + 1) * len(x_list), 1], y)
    ax.axvline(experiment_x[0], color='b', label='experiment')
    ax.axvline(experiment_x[1], color='b')
    for p in [0.688, 0.813, 0.938, 1.0,  1.125, 1.25, 1.375]:
        ax.axvline(p, linestyle=':', color='r')
    ax.axvline(x_normal[0], linestyle='--', color='b', label='nominal')
    ax.axvline(x_normal[1], linestyle='--', color='b')
    ax.grid(True)
    # ax.axis(ymin=0, ymax=0.13, xmin=-0.25, xmax=1.4)
    ax.legend(bbox_to_anchor=[-0.8, 1], loc='upper left',
              frameon=False,
              labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
              fancybox=False, shadow=False, )
    fig.subplots_adjust(left=0.44, right=0.98, bottom=0.15, top=0.97)  # presentation

    fig.savefig(os.path.join(plot_folder, 'separation_map'))
    plt.close('all')


sum_stat = ['cp', 'u', 'uv', 'cp_u', 'cp_u_uv', 'all_if_less_05']
folders = [os.path.join(output_folder, f"calibration_job{i}") for i in range(len(sum_stat))]

# cp_maps = np.empty((0, len(cp_nominal)))
# u_maps = np.empty((0, len(u_nominal)))
# uv_maps = np.empty((0, len(u_nominal)))
# x_maps = np.empty((0, 2))

# for i, folder in enumerate(folders):
#     cp_maps = np.vstack((cp_maps, np.fromfile(os.path.join(folder, 'cp_all.bin')).reshape(-1, 721)))
#     u_maps = np.vstack((u_maps, np.fromfile(os.path.join(folder, 'u_slice.bin')).reshape(-1, 800)))
#     uv_maps = np.vstack((uv_maps, np.fromfile(os.path.join(folder, 'uv_slice.bin')).reshape(-1, 800)))
#     with open(os.path.join(folder, 'result.dat')) as f:
#         lines = f.readlines()
#         for line in lines:
#             d = np.fromstring(line[1:-1], dtype=float, sep=',')
#             x_maps = np.vstack((x_maps, d[-3:-1].reshape(-1, 2)))

# plot_separation(x_maps)

# for i, stat in enumerate(sum_stat):
#     ind = i*len(sum_stat) + np.arange(len(x_list))
#     plot_cp_percent(cp_maps[ind], stat)
#     plot_u_subplot_percent(u_maps[ind], stat)
#     plot_uv_subplot_percent(uv_maps[ind], stat)

# for j, x in enumerate(x_list):
#     p = np.array([j+i*len(x_list) for i in range(len(sum_stat))])
#     plot_cp(cp_maps[p], x)
#     plot_u_subplot(u_maps[p], x)
#     plot_uv_subplot(uv_maps[p], x)
#     for slice in range(8):
#         # print(x_list[j], slice, '\t', uv_maps[p, slice * 100:(slice + 1) * 100][0])
#         plot_u(u_maps[p, slice * 100:(slice + 1) * 100], slice, x)
#         plot_uv(uv_maps[p, slice * 100:(slice + 1) * 100], slice, x)

cp_maps = np.fromfile(os.path.join(output_folder, 'cp_all.bin')).reshape(-1, 721)
u_maps =  np.fromfile(os.path.join(output_folder, 'u_slice.bin')).reshape(-1, 800)
uv_maps = np.fromfile(os.path.join(output_folder, 'uv_slice.bin')).reshape(-1, 800)

print(uv_maps.shape)
for slice in range(8):
    print(slice, '\t', uv_maps[0, slice * 100:(slice + 1) * 100][0])
    plot_u(u_maps[:, slice * 100:(slice + 1) * 100], slice, x)
    plot_uv(uv_maps[:, slice * 100:(slice + 1) * 100], slice, x)
plot_cp(cp_maps, x)