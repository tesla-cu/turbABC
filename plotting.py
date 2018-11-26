import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use('dark_background')

fig_width_pt = 1.5*246.0  # Get this from LaTeX using "The column width is: \the\columnwidth \\"
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

folder = './plots/'

# Load in validation data
folder_data = './Test/'
axi_exp_k = np.loadtxt(os.path.join(folder_data, 'axi_exp_k.txt'))
axi_exp_b = np.loadtxt(os.path.join(folder_data, 'axi_exp_b.txt'))
axi_con_k = np.loadtxt(os.path.join(folder_data, 'axi_con_k.txt'))
axi_con_b = np.loadtxt(os.path.join(folder_data, 'axi_con_b.txt'))
shear_k = np.loadtxt(os.path.join(folder_data, 'shear_k.txt'))
plane_k = np.loadtxt(os.path.join(folder_data, 'plane_k.txt'))
plane_b = np.loadtxt(os.path.join(folder_data, 'plane_b.txt'))


def plot(x1, y1, x2, y2, x3, y3, x4, y4):
    fig = plt.figure(figsize=(fig_width, 1.5*fig_height))
    ax = plt.gca()

    ax.plot(x1, y1, label='axisymmetric expansion')
    ax.scatter(axi_exp_k[:, 0], axi_exp_k[:, 1], marker='^')
    ax.plot(x2, y2, label='axisymmetric contraction')
    ax.scatter(axi_con_k[:, 0], axi_con_k[:, 1], marker='>')
    ax.plot(x3, y3, label='pure shear')
    ax.scatter(shear_k[:, 0], shear_k[:, 1], marker='o')
    ax.plot(x4, y4, label='plain strain')
    ax.scatter(plane_k[:, 0], plane_k[:, 1], marker='<')

    ax.set_xlabel(r'$S\cdot t$')
    ax.axis(xmin=0, xmax=5, ymin=0, ymax=2.5)
    plt.legend()
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(folder, 'test'))
    plt.close('all')


def plot_marginal_smooth_pdf(path, C_limits):

    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']
    N_params = len(C_limits)
    max_value = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i)+str(j)] = np.loadtxt(os.path.join(path['output'], 'marginal_smooth{}{}'.format(i, j)))
                max_value = max(max_value, np.max(data[str(i)+str(j)]))
    max_value = int(max_value)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmaplist[0] = 'black'   # force the first color entry to be black
    # cmaplist[0] = 'white' # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    fig = plt.figure(figsize=(1.25*fig_width, 1.1*fig_width))
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(path['output'], 'marginal_smooth{}'.format(i)))
                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                c_final_smooth = np.loadtxt(os.path.join(os.path.join(path['output'], 'C_final_smooth')))
                if len(c_final_smooth.shape) == 1:
                    ax.axvline(c_final_smooth[i], linestyle='--', color='b', label='max of joint pdf')
                elif len(c_final_smooth) < 4:
                    for C in c_final_smooth:
                        ax.axvline(C[i], linestyle='--', color='b', label='joint max')
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)

                if i == 0:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.03))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                if i == 1:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                if i == 2:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                if i == 3:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_locator(plt.NullLocator())
                # ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)
                ax.set_xlabel(params_names[i], labelpad=2)
                if i == 0:
                    if N_params == 3:
                        ax.legend(bbox_to_anchor=(2.35, -1.5), fancybox=True)
                    elif N_params == 4:
                        ax.legend(bbox_to_anchor=(3, -2.75), fancybox=True)
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])

                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_tick_params(direction='in')

                ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])

                im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
    cax = plt.axes([0.05, 0.1, 0.01, 0.26])
    plt.colorbar(im, cax=cax)   #, ticks=np.arange(max_value+1))

    if N_params == 3:
        # fig.subplots_adjust(left=0.02, right=0.9, wspace=0.1, hspace=0.1, bottom=0.1, top=0.98)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.28, hspace=0.1, bottom=0.1, top=0.98)
    elif N_params == 4:
        fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3, hspace=0.1, bottom=0.1, top=0.98)
    elif N_params == 6:
        fig.subplots_adjust(left=0.05, right=0.98, wspace=0.45, hspace=0.35, bottom=0.08, top=0.98)

    fig.savefig(os.path.join(path['plots'], 'marginal_smooth'))
    plt.close('all')

