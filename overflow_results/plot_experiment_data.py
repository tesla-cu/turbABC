import os
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from numpy.linalg import norm as norm2

basefolder = './'
data_folder = os.path.join(basefolder, 'data/')
data_folder2= os.path.join(basefolder, 'data_extraction/output/')
plot_folder = os.path.join(basefolder, 'plots/')


def calc_dist_norm2(x, y, valid_data_x, valid_data_y):

    points = np.interp(valid_data_x, x, y)
    # points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))   # adding gaussian noise
    diff = norm2(points - valid_data_y)
    return diff


def readfile_with_zones(filename):
    with open(filename) as f:
        text = f.readlines()
    x, ind, array = [], [], []
    for line in text:
        if line[:4] == 'ZONE':
            ind.append(len(array))
            x.append(line[-8:-2])
        else:
            array.append([float(i) for i in line.split()])
    ind.append(len(array))
    experiment = [array[ind[i]:ind[i + 1]] for i in range(len(ind) - 1)]
    return x, experiment


def load_grid_y(folder):

    size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
    grid_y = np.fromfile(os.path.join(folder, 'grid_y'))
    grid_y = np.rollaxis(grid_y.reshape((size[2], size[0])), 1)
    indices = np.flip(np.loadtxt(os.path.join(folder, 'indices'), dtype=np.int)) - 1
    grid_y = grid_y[indices, :]
    grid_y = grid_y - grid_y[:, 0].reshape(-1, 1)
    return grid_y


def load_grid(folder):

    size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
    grid_y = np.fromfile(os.path.join(folder, 'grid_y'))
    grid_y = np.rollaxis(grid_y.reshape((size[2], size[0])), 1)
    grid_x = np.fromfile(os.path.join(folder, 'grid_x'))
    grid_x = np.rollaxis(grid_x.reshape((size[2], size[0])), 1)

    return grid_x, grid_y


def load_indices(folder):
    indices = np.flip(np.loadtxt(os.path.join(folder, 'indices'), dtype=np.int)) - 1
    return indices


def load_grid_x(folder):

    size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
    grid_x = np.fromfile(os.path.join(folder, 'grid_x'))
    grid_x = np.rollaxis(grid_x.reshape((size[2], size[0])), 1)

    return grid_x[:, 0]


def plot_u(data, exp_folder, grid_folder):

    y = load_grid_y(grid_folder)
    indices = load_indices(grid_folder)
    fig = plt.figure()
    ax = plt.gca()
    x, experiment = readfile_with_zones(os.path.join(exp_folder, 'experiment_u.txt'))
    for x_position, x_label in enumerate(indices):
        exp = np.array(experiment[x_position])
        # cfl3d = np.array(cfl3d_u[x_position])[:-1]
        ax.scatter(exp[:, 0], exp[:, 1], linewidth=2, label=r'$x/c$ = '+str(x_label))
        # ax.plot(cfl3d[:, 0], cfl3d[:, 1], '-', linewidth=2, label=r'cfl3D')
        ax.plot(data[x_position], y[x_position], '-', linewidth=2)
    ax.grid(True)
    ax.set_xlabel(r'$\overline{u}/U_{\mathrm{inf}}$')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.axis(xmin=-0.2, xmax=1.4, ymin=0, ymax=0.14)
    ax.set_ylabel(r'$y^{\prime}/c$')
    plt.legend(loc=0)
    fig.savefig(os.path.join(plot_folder, 'u_test'))
    plt.close('all')


def plot_uv(data, exp_folder, grid_folder):

    y = load_grid_y(grid_folder)
    indices = load_indices(grid_folder)
    fig = plt.figure()
    ax = plt.gca()
    x, experiment = readfile_with_zones(os.path.join(exp_folder, 'experiment_uv.txt'))
    for x_position, x_label in enumerate(indices):
        exp = np.array(experiment[x_position])
        # cfl3d = np.array(cfl3d_u[x_position])[:-1]
        ax.scatter(exp[:, 0], exp[:, 1], linewidth=2, label=r'$x/c$ = '+str(x_label))
        # ax.plot(cfl3d[:, 0], cfl3d[:, 1], '-', linewidth=2, label=r'cfl3D')
        ax.plot(-data[x_position], y[x_position], '-', linewidth=2)
    ax.grid(True)
    ax.set_xlabel(r'$u^{\prime}v^{\prime}/U_{\mathrm{inf}}$')
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.axis(xmin=-0.02, xmax=0, ymin=0, ymax=0.14)
    ax.set_ylabel(r'$y^{\prime}/c$')
    plt.legend(loc=0)
    fig.savefig(os.path.join(plot_folder, 'uv_test'))
    plt.close('all')


def plot_cp(data, exp_folder, grid_folder):
    experiment = np.loadtxt(os.path.join(exp_folder, 'experiment_cp.txt'))
    x_min, x_max = np.min(experiment[:, 0]), np.max(experiment[:, 0])
    # Cp cfl3D data
    cfl3d = np.loadtxt(os.path.join(exp_folder, 'cfl3d_cp.txt'))
    ind_cfl3d = np.where(np.logical_and(cfl3d[:, 0] > x_min, cfl3d[:, 0] < x_max))[0]
    # Cp data from overflow
    grid_x = load_grid_x(grid_folder)
    ind = np.where(np.logical_and(grid_x > x_min, grid_x < x_max))[0]
    x = grid_x[ind]
    fig = plt.figure()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.plot(x, data[ind])
    ax.plot(experiment[:, 0], experiment[:, 1], 'r+', linewidth=2, label='experiment')
    # ax.plot(x, cp_array[4], style[4], linewidth=lw[4], label='overflow nominal')
    ax.plot(cfl3d[ind_cfl3d, 0], cfl3d[ind_cfl3d, 1], linewidth=2, label='CFL3D')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$C_p$')
    plt.legend(loc=0)
    fig.savefig(os.path.join(plot_folder, 'cp_test'))
########################################################################
# Cp experimental data
########################################################################
# experiment

# ########################################################################################################################
# #
# ########################################################################################################################
#
# dist = [calc_dist_norm2(x, data[ind, 2], experiment[:, 0], experiment[:, 1])]
# cp_array = np.empty((6, len(ind)))
# cp_array[0] = data[ind, 2]
# for i in range(1, 6):
#     data = np.loadtxt(os.path.join(data_folder, 'cp.dat/cp.dat.{}'.format(i)))
#     cp_array[i] = data[ind, 2]
#     dist.append(calc_dist_norm2(x, data[ind, 2], experiment[:, 0], experiment[:, 1]))
#
# labels = [r'$\beta*_{left}$', r'$\beta*_{right}$',
#           r'$\sigma_{\omega1\ left}$', r'$\sigma_{\omega1\ right}$',
#           r'$a_{1\ left} = nominal$', r'$a_{1\ right}$']
# style = ['--', '--', '--', '--', '-', '--']
# lw = [1, 1, 1, 1, 2, 1]
# fig = plt.figure()
# ax = plt.gca()
# ax.invert_yaxis()
# for i in range(6):
#     ax.plot(x, cp_array[i], style[i], linewidth=lw[i], label=labels[i])
# ax.plot(experiment[:, 0], experiment[:, 1], 'r+', linewidth=2, label='experiment')
# # ax.plot(x, cp_array[4], style[4], linewidth=lw[4], label='overflow nominal')
# # ax.plot(cfl3d[ind_cfl3d, 0], cfl3d[ind_cfl3d, 1], linewidth=lw[4], label='CFL3D')
# ax.grid(True)
# ax.set_xlabel('x')
# ax.set_ylabel(r'$C_p$')
# plt.legend(loc=0)
# fig.savefig(os.path.join(plot_folder, 'cp'))
# plt.close('all')
# ########################################################################################################################
# #
# ########################################################################################################################
# uv experimental data
x, experiment = readfile_with_zones(os.path.join(data_folder, 'experiment_uv.txt'))

fig = plt.figure()
ax = plt.gca()
for i, x_label in enumerate(x):
    exp = np.array(experiment[i])
    ax.plot(exp[:, 0], exp[:, 1], '-o', linewidth=2, label=r'$x/c$ = '+x_label)
# ax.plot(x, cp_array[4], style[4], linewidth=lw[4], label='overflow nominal')
# ax.plot(cfl3d[ind_cfl3d, 0], cfl3d[ind_cfl3d, 1], linewidth=lw[4], label='CFL3D')
ax.grid(True)
ax.set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\mathrm{inf}}$')
ax.xaxis.set_major_locator(MultipleLocator(0.005))
ax.set_ylabel(r'$y^{\prime}/c$')
plt.legend(loc=0)
fig.savefig(os.path.join(plot_folder, 'uv_experiment'))
plt.close('all')


data = np.fromfile(os.path.join(data_folder, 'uv'), dtype='float32')[:-1]
fig = plt.figure()
ax = plt.gca()
exp = np.array(experiment[0])
ax.plot(exp[:, 0], exp[:, 1], '-o', linewidth=2, label=r'experiment')
# ax.plot(data, data_y, '-o', linewidth=2, label=r'overflow')
# ax.plot(x, cp_array[4], style[4], linewidth=lw[4], label='overflow nominal')
# ax.plot(cfl3d[ind_cfl3d, 0], cfl3d[ind_cfl3d, 1], linewidth=lw[4], label='CFL3D')
ax.grid(True)
ax.set_xlabel(r'$\overline{u^{\prime}v^{\prime}}/U^2_{\mathrm{inf}}$')
ax.xaxis.set_major_locator(MultipleLocator(0.005))
ax.axis(xmin=-0.02, xmax=0, ymin=0, ymax=0.14)
ax.set_ylabel(r'$y^{\prime}/c$')
plt.legend(loc=0)
fig.savefig(os.path.join(plot_folder, 'uv'))
plt.close('all')
# ########################################################################################################################
# #
# ########################################################################################################################
# u experimental data
x, experiment = readfile_with_zones(os.path.join(data_folder, 'experiment_u.txt'))
# u from cfl3d
_, cfl3d_u = readfile_with_zones(os.path.join(data_folder, 'cfl3d_u.txt'))
fig = plt.figure()
ax = plt.gca()
for i, x_label in enumerate(x):
    exp = np.array(experiment[i])
    cfl = np.array(cfl3d_u[i])
    ax.scatter(exp[:, 0], exp[:, 1], linewidth=2, label=r'$x/c$ = '+x_label)
    ax.plot(cfl[:, 0], cfl[:, 1])
ax.grid(True)
ax.set_xlabel(r'$\overline{u}/U^2_{\mathrm{inf}}$')
ax.set_ylabel(r'$y^{\prime}/c$')
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.axis(xmin=-0.5, xmax=1.5, ymin=0, ymax=0.14)
plt.legend(loc=0)
fig.savefig(os.path.join(plot_folder, 'u_experiment'))
plt.close('all')


data = np.fromfile(os.path.join(data_folder2, 'u'))
data = data.reshape((size[2], len(indices)))
data = np.flip(np.rollaxis(data, 1), 0)
y = grid_y[-indices, :]
y = y - y[:, 0].reshape(-1, 1)
fig = plt.figure()
ax = plt.gca()
for x_position, x_label in enumerate(x):
    exp = np.array(experiment[x_position])
    cfl3d = np.array(cfl3d_u[x_position])[:-1]
    ax.scatter(exp[:, 0], exp[:, 1], linewidth=2, label=r'$x/c$ = '+x_label)
    # ax.plot(cfl3d[:, 0], cfl3d[:, 1], ':', linewidth=2)
    ax.plot(data[x_position], y[x_position], '-', linewidth=2)
ax.grid(True)
ax.set_xlabel(r'$\overline{u}/U_{\mathrm{inf}}$')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.axis(xmin=-0.2, xmax=1.4, ymin=0, ymax=0.14)
ax.set_ylabel(r'$y^{\prime}/c$')
plt.legend(loc=0)
fig.savefig(os.path.join(plot_folder, 'u'))
plt.close('all')