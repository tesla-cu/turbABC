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
from rans_ode.sumstat import TruthData
# plt.style.use('dark_background')



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

folder = './plots/'


def plot_marginal_smooth_pdf(data_folder, C_limits, num_bin_joint, params_names, plot_folder):

    N_params = len(C_limits)
    max_value, max_value2 = 0.0, 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i) + str(j)] = np.loadtxt(os.path.join(data_folder, 'marginal_smooth{}{}'.format(i, j)))
                max_value = max(max_value, np.max(data[str(i)+str(j)]))
            if i > j:
                data[str(i) + str(j)] = np.loadtxt(
                    os.path.join(data_folder, 'conditional_smooth{}{}'.format(i, j)))
                norm = np.sum(data[str(i) + str(j)])
                # print('norm = ', norm, np.max(data[str(i) + str(j)]))
                data[str(i) + str(j)] /= norm
                max_value2 = max(max_value2, np.max(data[str(i) + str(j)]))
            # print(max_value, max_value2)

    # max_value = int(max_value)
    # cmap = plt.cm.jet  # define the colormap
    # cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    # # cmaplist[0] = 'black'   # force the first color entry to be black
    # cmaplist[0] = 'white' # force the first color entry to be white
    # cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    ###################################################################################################
    cmap2 = plt.cm.inferno  # define the colormap
    cmap = plt.cm.BuPu  # define the colormap
    cmaplist = [cmap(i) for i in reversed(range(cmap.N))]  # extract all colors from the map
    cmaplist2 = [cmap2(i) for i in (range(cmap2.N))]  # extract all colors from the map
    gamma1 = 1
    gamma2 = 1
    ###################################################################################################
    # cmap2 = plt.cm.Greys   # define the colormap
    # cmap = plt.cm.Greys  # define the colormap
    # cmaplist = [cmap(i) for i in (range(cmap.N))]  # extract all colors from the map
    # cmaplist2 = [cmap2(i) for i in (range(cmap2.N))]  # extract all colors from the map
    # gamma1 = 1
    # gamma2 = 1
    ###################################################################################################
    # cmaplist[0] = 'black'  # 'white' # force the first color entry to be white
    # cmaplist2[0] = 'black'  # 'white' # force the first color entry to be white
    cmaplist[0] = 'white' # force the first color entry to be white
    cmaplist2[0] = 'white'  # force the first color entry to be white
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist)
    cmap2 = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist2)

    width, height = fig_size(double_column)
    confidence = np.loadtxt(os.path.join(data_folder, 'confidence_75'))
    print(width, height)
    fig = plt.figure(figsize=(width, height))
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(data_folder, 'marginal_smooth{}'.format(i)))
                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                c_final_smooth = np.loadtxt(os.path.join(data_folder, 'C_final_smooth{}'.format(num_bin_joint)))
                ax.axvline(confidence[i, 0], linestyle='--', color='b', label=r'$75\%$ interval')
                ax.axvline(confidence[i, 1], linestyle='--', color='b')
                if len(c_final_smooth.shape) == 1:
                    ax.axvline(c_final_smooth[i], linestyle='--', color='r', label='max of joint pdf')
                elif len(c_final_smooth) < 4:
                    for C in c_final_smooth:
                        ax.axvline(C[i], linestyle='--', color='r', label='joint max')
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)

                # if i == 0:
                #     ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # if i == 1:
                #     ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # if i == 2:
                #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # if i == 3:
                #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_locator(plt.NullLocator())
                if i != 3:
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_xlabel(params_names[i], labelpad=2)
                ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)

                # if i == 0:
                #     ax.legend(bbox_to_anchor=(3, -2.75), fancybox=True)
                #     textstr = '\n'.join((
                #         r'$C_1=%.3f$' % (c_final_smooth[0],),
                #         r'$C_2=%.3f$' % (c_final_smooth[1],),
                #         r'$C_{\epsilon1}=%.3f$' % (c_final_smooth[2],),
                #         r'$C_{\epsilon2}=%.3f$' % (c_final_smooth[3],)))
                #     ax.text(0.15, -1.6, textstr, transform=ax.transAxes, fontsize=12,
                #             verticalalignment='top', linespacing=1.5)
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
                if j == 3:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(params_names[i], labelpad=2)
                else:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())

                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_tick_params(direction='in')

                ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)

                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])

                im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
            elif i > j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])
                ax.tick_params(axis='both', which='major', pad=2)
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                # if True and j != 2:
                #     ax.text(0.02, 0.07, '(' + string.ascii_lowercase[i * N_params + j] + ')',
                #             transform=ax.transAxes, size=10, weight='black')
                # else:
                #     ax.text(0.02, 0.85, '('+string.ascii_lowercase[i*N_params+j]+')',
                #         transform=ax.transAxes, size=10, weight='black')
                if j == 0:
                    ax.set_ylabel(params_names[i])
                else:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                if i != (N_params - 1):
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_xlabel(params_names[j], labelpad=2)

                im_cond = ax.imshow(data[str(i) + str(j)], origin='lower', cmap=cmap2, aspect='auto', extent=ext,
                                    norm=colors.PowerNorm(gamma=gamma2), vmax=max_value2)
                ax.axvline(c_final_smooth[j], linestyle='--', color='r')
                ax.axhline(c_final_smooth[i], linestyle='--', color='r')
    # cax = plt.axes([0.05, 0.1, 0.01, 0.26])
    # plt.colorbar(im, cax=cax)   #, ticks=np.arange(max_value+1))

    fig.subplots_adjust(left=0.20, right=0.80, wspace=0.1, hspace=0.1, bottom=0.1, top=0.98)
    fig.savefig(os.path.join(plot_folder, 'marginal_smooth'))
    plt.close('all')