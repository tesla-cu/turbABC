import os
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

fig_width_pt = 1.5*246.0  # Get this from LaTeX using "The column width is: \the\columnwidth \\"
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

N_params = 4

path = '../ABC/num_bins/{}_params/'.format(N_params)

titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$', r'$\sigma_{ij}S_{ij}$']


fig1, axarr1 = plt.subplots(nrows=1, ncols=N_params, figsize=(2*N_params, 2.5))
fig2, axarr2 = plt.subplots(nrows=1, ncols=N_params, figsize=(2*N_params, 2.5))

if N_params == 3:
    C_limits = [[-0.15, 0],
                [-0.25,  0.25],
                [-0.14,  0.14]]
    num = np.arange(10, 50)
elif N_params ==4:
    C_limits = [[-0.09, 0],
                [-0.15,  0.15],
                [-0.18,  0.18],
                [-0.18, 0.18]]
    num = np.arange(10, 25)


for num_bin_joint in num:
    x1 = np.loadtxt(os.path.join(path, 'C_final_joint' + str(num_bin_joint)))
    x2 = np.loadtxt(os.path.join(path, 'C_final_smooth' + str(num_bin_joint)))
    if len(x1.shape)>1:
        print(x1.shape)
        print('{}\n{}\n{}\n'.format(num_bin_joint, x1, x2))
        for i in range(N_params):
            axarr1[i].scatter([num_bin_joint]*x1.shape[0], x1[:, i])
    else:
        for i in range(N_params):
            axarr1[i].scatter([num_bin_joint], x1[i])
    if len(x2.shape)>1:
        print(x1.shape)
        print('{}\n{}\n{}\n'.format(num_bin_joint, x1, x2))
        for i in range(N_params):
            axarr2[i].scatter([num_bin_joint]*x2.shape[0], x2[:, i])
    else:
        for i in range(N_params):
            axarr2[i].scatter([num_bin_joint], x2[i])

for i in range(N_params):
    axarr1[i].set_ylim(C_limits[i])
    axarr2[i].set_ylim(C_limits[i])
    axarr1[i].set_xlabel('Number of bins')
    axarr2[i].set_xlabel('Number of bins')

axarr1[0].set_ylabel(r'$C_i$')
axarr2[0].set_ylabel(r'$C_i$')

fig1.subplots_adjust(left=0.1, right=0.98, hspace=0.3, bottom=0.21, top=0.97)
fig2.subplots_adjust(left=0.1, right=0.98, hspace=0.3, bottom=0.21, top=0.97)

fig1.savefig(os.path.join(path, 'Num_bins_joint_{}'.format(N_params)))
fig2.savefig(os.path.join(path, 'Num_bins_smooth_{}'.format(N_params)))
plt.close('all')