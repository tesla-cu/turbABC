import numpy as np
import os
import string
import sys
sys.path.append('/Users/olgadorr/Research/ABC_MCMC')
from scipy.io import FortranFile

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

import pyabc.glob_var as g
from overflow.sumstat import TruthData, calc_sum_stat
from overflow.overflow_driver import Overflow
import overflow.sumstat as sumstat

exp_folder = '../overflow/valid_data/'
nominal_folder = '/Users/olgadorr/Research/ABC_MCMC/nominal/'
plot_folder = '../plots_nominal/'
scott_folder = '../overflow_scott/cp.dat/'
job_folder = nominal_folder


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


Truth = TruthData(exp_folder, ['cp', 'u', 'uv'])
sumstat_true = Truth.sumstat_true

# fortran_cp = np.loadtxt(os.path.join(nominal_folder, 'cp.dat'))

with open(os.path.join(nominal_folder, 'result.dat')) as f:
    line = f.readline()
    result = np.fromstring(line[1:-1], dtype=float, sep=',')[5:]

overflow = Overflow(job_folder, exp_folder, exp_folder, 1)
g.job_folder = job_folder
g.Grid = sumstat.GridData(exp_folder)


cp, u, uv, _, _ = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.x_slices, g.Grid.y_slices)


cp_slice = np.fromfile(os.path.join(nominal_folder, 'cp_all.bin'), dtype=float)
print(cp_slice.shape)
u_slices = np.fromfile(os.path.join(nominal_folder, 'u_slice.bin'), dtype=float)
uv_slices = np.fromfile(os.path.join(nominal_folder, 'uv_slice.bin'), dtype=float)

cp_scott2 = np.loadtxt(os.path.join(scott_folder, 'cp.dat.2'))

### CFL3D results ####################################################################
cp_cfl3d = np.loadtxt(os.path.join(exp_folder, 'cfl3d_cp.txt'))
x_cfl, _, u_cfl3d = sumstat.readfile_with_zones(os.path.join(exp_folder, 'cfl3d_u.txt'))
_, _, uv_cfl3d = sumstat.readfile_with_zones(os.path.join(exp_folder, 'cfl3d_uv.txt'))

#############################################################################
names = [r'$x/c = -0.25$', r'$x/c = 0.688$', r'$x/c = 0.813$', r'$x/c = 0.938$',
         r'$x/c = 1.0$', r'$x/c = 1.125$', r'$x/c = 1.25$', r'$x/c = 1.375$', 'overflow nominal']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors[8] = 'k'
markers = ['o', 'v', '^', '<', '>', 's', 'p', 'D', '']
lss = ['--']*9
lss[8] = '-'
print(markers)
print(colors)
custom_lines = [Line2D([0], [0], marker=markers[i], linestyle=lss[i], fillstyle='none', markersize=4,
                       color=colors[i], lw=1) for i in range(9)]


def read_data_from_overflow_old(job_folder, grid, indices):
    ########################################################################
    # Read data
    ########################################################################
    f = FortranFile(os.path.join(job_folder, 'q.save'), 'r')
    f.read_ints(np.int32)[0]  # ng: number of geometries
    (jd, kd, ld, nq, nqc) = tuple(f.read_ints(np.int32))
    type = np.array([np.dtype('<f8')] * 16)
    type[7] = np.dtype('<i4')
    (fm, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = tuple(f.read_record(*type))
    # fm, a, re, t, gamma, beta, tinf, igamma, htinf, ht1, ht2, rgas1, rgas2, refmach, tvref, dtvref))
    data = f.read_reals(dtype=np.float64).reshape((nq, ld, kd, jd))
    data = data[:, :, 1, :]  # taking 2D data (middle in y direction)
    ###################
    u = np.rollaxis(data[1] / data[0], 1)
    ###################
    surface_data = data[:, 0, :]
    v2 = 0.5 * (surface_data[1] ** 2 + surface_data[2] ** 2 + surface_data[3] ** 2) / surface_data[0]
    p = (surface_data[5] - 1.0) * (surface_data[4] - v2)
    cp = (p - 1.0 / surface_data[5]) / (0.5 * fm * fm)
    ###################
    x_grid, y_grid = grid
    dUdy = np.empty((jd, ld))
    for j in range(jd):
        dUdy[j] = np.gradient(u[j], y_grid[j])
    uv = np.rollaxis(data[6] / data[7], 1) * dUdy
    return cp, u[indices], uv[indices]


# _, u_old, uv_old = read_data_from_overflow_old(g.job_folder, g.Grid.grid, g.Grid.indices)


def plot_cp():
    experiment_cp = Truth.cp
    x_min, x_max = np.min(experiment_cp[:, 0]), np.max(experiment_cp[:, 0])
    # ind = np.where(np.logical_and(g.Grid.grid_x > x_min, g.Grid.grid_x < x_max))[0]
    # overflow = cp
    # cp = calc_sum_stat(g.Grid.grid_x[::-1], cp[::-1], Truth.cp[:, 0])
    # Cp data from overflow
    overflow_cp = result[:Truth.length[0]]
    # fortran
    # ind2 = np.where(np.logical_and(fortran_cp[:, 0] > x_min, fortran_cp[:, 0] < x_max))[0]
    # f_cp = fortran_cp[ind2, 2]

    width, height = fig_size(oneandhalf_column)
    fig = plt.figure(figsize=(width, 1.25*height))
    ax = plt.gca()
    ax.invert_yaxis()
    # ax.plot(g.Grid.grid_x, overflow, label='overflow')
    ax.plot(g.Grid.grid_x, cp_slice, label='overflow')
    ax.plot(cp_cfl3d[:, 0], cp_cfl3d[:, 1], label='CFL3D')
    # ax.plot(experiment_cp[:, 0], cp, label='overflow interpolate')
    # ax.plot(experiment_cp[:, 0], overflow_cp, label='sumstat nominal')
    # ax.plot(fortran_cp[:, 0], fortran_cp[:, 2], label='fortran nominal')
    # ax.plot(cp_scott2[:, 0], cp_scott2[:, 2], label='scott_2')
    ax.plot(experiment_cp[:, 0], experiment_cp[:, 1], 'r+', linewidth=3, label='experiment')
    # ax.plot(cfl3d[ind_cfl3d, 0], cfl3d[ind_cfl3d, 1], linewidth=2, label='CFL3D')
    ax.grid(True)
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$C_p$')
    ax.axis(xmin=0.4, xmax=1.8, ymin=0.2, ymax=-0.9)
    plt.legend(loc=0)
    # fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.9)

    fig.savefig(os.path.join(plot_folder, 'cp_experiment'))


def plot_u_uv_experiment():
    y = np.linspace(0, 0.14, 100)
    experiment_u = Truth.u
    experiment_uv = Truth.uv
    width, height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(width, 0.9*height))
    # u plot #######################
    i = 0
    for j, exp_u in enumerate(experiment_u):
        # axarr[0].plot(u[i:i+len(exp_u)], g.Grid.x_slices[i:i+len(exp_u)], 'o', color=colors[j], label='overflow')
        axarr[0].plot(u_slices[j*100:(j+1)*100], y, '-', color=colors[j], label='overflow')
        axarr[0].plot(exp_u[:, 0], exp_u[:, 1],  marker=markers[j], markersize=3.5, fillstyle='none',
                      ls=lss[j], color=colors[j], linewidth=1)
        i += len(exp_u)
        axarr[0].grid(True)
        axarr[0].set_xlabel(r'$\overline{u}/U_{\inf}$')
        axarr[0].set_ylabel(r'$y/c$')
        axarr[0].axis(ymin=0, ymax=0.14, xmin=-0.25, xmax=1.5)
        axarr[0].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axarr[0].text(0.05, 0.92,  'a)', transform=axarr[0].transAxes, size=11, weight='bold')
        # u plot #######################
    i = 0
    for j, exp_uv in enumerate(experiment_uv):
        # axarr[1].plot(-uv[i:i+len(exp_uv)], g.Grid.x_slices[i:i+len(exp_uv)], 'o', color=colors[j], label='overflow')
        axarr[1].plot(-uv_slices[j*100:(j+1)*100], y, '-', color=colors[j], label='overflow')
        axarr[1].plot(exp_uv[:, 0], exp_uv[:, 1], marker=markers[j], markersize=3.5, fillstyle='none',
                      ls=lss[j], color=colors[j], linewidth=1)
        i += len(exp_uv)
        axarr[1].grid(True)
        axarr[1].set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\mathrm{inf}}$')
        axarr[1].axis(ymin=0, ymax=0.14, xmin=-0.02, xmax=0)
        axarr[1].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axarr[1].text(0.05, 0.92, 'b)', transform=axarr[1].transAxes, size=11, weight='bold')
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.1, bottom=0.12, top=0.82)
    axarr[0].legend(custom_lines, names,
                    ncol=3, loc='upper center', bbox_to_anchor=[1.05, 1.29], labelspacing=0.0,
                    handletextpad=0.5, handlelength=1.5, fancybox=False, shadow=False, frameon=False)
    fig.savefig(os.path.join(plot_folder, 'u_uv_experiment'))


def plot_u_uv_cfl3d():
    y = np.linspace(0, 0.14, 100)
    experiment_u = Truth.u
    experiment_uv = Truth.uv
    width, height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(width, 0.9*height))
    # u plot #######################
    i = 0
    for j, exp_u in enumerate(experiment_u):
        # axarr[0].plot(u_old[j], g.Grid.grid_y[j], ':', color=colors[j], label='overflow no interp')
        axarr[0].plot(u_cfl3d[j][:, 0], u_cfl3d[j][:, 1], '--', color=colors[j], label='CFL3D')
        axarr[0].plot(u_slices[j*100:(j+1)*100], y, '-', color=colors[j], label='overflow')
        i += len(exp_u)
        axarr[0].grid(True)
        axarr[0].set_xlabel(r'$\overline{u}/U_{\inf}$')
        axarr[0].set_ylabel(r'$y/c$')
        axarr[0].axis(ymin=0, ymax=0.14, xmin=-0.25, xmax=1.5)
        axarr[0].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axarr[0].text(0.05, 0.92,  'a)', transform=axarr[0].transAxes, size=11, weight='bold')
        # u plot #######################
    i = 0
    for j, exp_uv in enumerate(experiment_uv):
        # axarr[1].plot(-uv_old[j], g.Grid.grid_y[j], ':', color=colors[j], label='overflow no interp')
        axarr[1].plot(-uv_slices[j*100:(j+1)*100], y, '-', color=colors[j], label='overflow')
        axarr[1].plot(uv_cfl3d[j][:, 0], uv_cfl3d[j][:, 1], '--', color=colors[j], label='CFL3D')
        i += len(exp_uv)
        axarr[1].grid(True)
        axarr[1].set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\mathrm{inf}}$')
        axarr[1].axis(ymin=0, ymax=0.14, xmin=-0.02, xmax=0)
        axarr[1].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axarr[1].text(0.05, 0.92, 'b)', transform=axarr[1].transAxes, size=11, weight='bold')
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.1, bottom=0.12, top=0.82)
    axarr[0].legend(custom_lines, names,
                    ncol=3, loc='upper center', bbox_to_anchor=[1.05, 1.29], labelspacing=0.0,
                    handletextpad=0.5, handlelength=1.5, fancybox=False, shadow=False, frameon=False)
    fig.savefig(os.path.join(plot_folder, 'u_uv_cfl3d'))


def plot_uv():
    experiment_uv = Truth.uv
    overflow_uv = result[Truth.length[1]:Truth.length[2]]
    fig = plt.figure(figsize=fig_size(single_column))
    ax = plt.gca()
    i = 0
    for j, exp_uv in enumerate(experiment_uv[:2]):
        # stat_uv = calc_sum_stat(g.Grid.grid_y[j], uv[j], Truth.uv[j][:, 1])
        # ax.plot(-stat_uv, Truth.uv[j][:, 1], '-o', color=colors[j], label='overflow interpolate')
        ax.plot(-uv[i:i + len(exp_uv)], g.Grid.x_slices[i:i + len(exp_uv), 1], '-', color=colors[j],
                      label='overflow')
        # ax.plot(-uv[j], g.Grid.grid_y[j], '-', color=colors[j], label='overflow')
        # ax.plot(-overflow_uv[i:i+len(exp_uv)], exp_uv[:, 1], ':', color=colors[j], label='sumstat overflow')
        ax.plot(exp_uv[:, 0], exp_uv[:, 1], '+', color=colors[j], linewidth=2, label=names[j])
        i += len(u)
    ax.grid(True)
    ax.set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\mathrm{inf}}$')
    ax.set_ylabel(r'$y/c$')
    ax.axis(ymin=0, ymax=0.14)
    plt.legend(loc=0)
    fig.savefig(os.path.join(plot_folder, 'uv_experiment'))









plot_cp()
plot_u_uv_experiment()
plot_u_uv_cfl3d()
# plot_uv()