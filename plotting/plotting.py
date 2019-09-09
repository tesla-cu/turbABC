import os
import numpy as np

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors

# plt.style.use('dark_background')

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
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size'] - 1
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


def plot_marginal_change(data_folders, params_names, C_limits, num_bin_kde, plot_folder):

    N_params = len(params_names)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(data_folders))])
    labels = []
    fig, axarr = plt.subplots(nrows=1, ncols=N_params, sharey=True, figsize=(fig_width, 0.8*fig_height))
    for folder in data_folders:
        x = os.path.basename(os.path.normpath(folder))[2:]
        eps = np.round(np.loadtxt(os.path.join(folder, 'eps')), 3)
        # labels.append('x = {}\%, eps = {}'.format(x, eps))
        MAP_x = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
        MAP_x = MAP_x.reshape((-1, N_params))
        labels.append('x = {}\%'.format(x))
        for i in range(N_params):
            data_marg = np.loadtxt(os.path.join(folder, 'marginal_smooth{}'.format(i)))
            for map in MAP_x:
                MAP_y = np.interp(map[i], data_marg[0], data_marg[1])
                axarr[i].scatter(map[i], MAP_y, color='r', s=10, zorder=2)
            axarr[i].plot(data_marg[0], data_marg[1], zorder=1)
            axarr[i].yaxis.set_major_formatter(plt.NullFormatter())
            axarr[i].set_xlabel(params_names[i])
            axarr[i].set_xlim(C_limits[i])
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

    fig.savefig(os.path.join(plot_folder, 'marginal_change'))
    plt.close('all')


def plot_MAP_confidence_change(data_folders, params_names, num_bin_kde, C_limits, plot_folder):
    N_params = len(params_names)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(data_folders))])
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
        MAP.append(np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde))))
        for j, level in enumerate(conf_level):
            confidence[i, :, j] = np.loadtxt(os.path.join(folder, 'confidence_{}'.format(int(100*(1-level)))))
            quantile[i, :, j] = np.loadtxt(os.path.join(folder, 'quantile_{}'.format(int(100*(1-level)))))
    ind = np.argsort(x)
    x = x[ind]
    MAP = np.array(MAP)[ind]
    confidence = confidence[ind]
    quantile = quantile[ind]
    colors = ['b', 'g', 'y']
    colors2 = ['orange', 'k', 'magenta']
    for i in range(N_params):
        for j in range(len(x)):
            MAP[j] = MAP[j].reshape((-1, N_params))
            for k in range(MAP[j].shape[0]):
                axarr[i].scatter(MAP[j][k, i], x[j], s=10, color='r', zorder=2)
        for j in range(len(conf_level)):
            axarr[i].semilogy(confidence[:, i, j,  0], x, color=colors[j])
            axarr[i].semilogy(confidence[:, i, j, 1], x, color=colors[j])
            axarr[i].semilogy(quantile[:, i, j,  0], x, color=colors2[j])
            axarr[i].semilogy(quantile[:, i, j, 1], x, color=colors2[j])
        axarr[i].set_xlabel(params_names[i])
        axarr[i].set_xlim(C_limits[i])

    axarr[0].set_ylabel('x [\%]')
    fig.subplots_adjust(left=0.12, right=0.98, wspace=0.1, hspace=0.1, bottom=0.2, top=0.8)

    custom_lines = [Line2D([0], [0], marker='o', color='r', lw=1)]
    legend_text = ['MAP']
    for j, level in enumerate(conf_level):
        custom_lines.append(Line2D([0], [0], color=colors[j], linestyle='-', lw=1))
        legend_text.append('{}\% confidence'.format(int(100*(1-level))))
    axarr[1].legend(custom_lines, legend_text, loc='upper center',
                    bbox_to_anchor=[0.7, 1.35], frameon=False,
                    labelspacing=0.0, handletextpad=0.5, handlelength=1.5,
                    fancybox=False, shadow=False, ncol=2)

    fig.savefig(os.path.join(plot_folder, 'MAP_change'))
    plt.close('all')


def plot_eps_change(data_folders, plot_folder):

    fig = plt.figure(figsize=(0.5*fig_width, 0.5*fig_height))
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


def plot_marginal_smooth_pdf(data_folder, C_limits, num_bin_joint, params_names, plot_folder):

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

    confidence = np.loadtxt(os.path.join(data_folder, 'confidence_75'))
    fig = plt.figure(figsize=(1.25*fig_width, 1.1*fig_width))
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
                # ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)
                ax.set_xlabel(params_names[i], labelpad=2)
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
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_tick_params(direction='in')

                ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])

                im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
    # cax = plt.axes([0.05, 0.1, 0.01, 0.26])
    # plt.colorbar(im, cax=cax)   #, ticks=np.arange(max_value+1))

    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3, hspace=0.1, bottom=0.1, top=0.98)
    fig.savefig(os.path.join(plot_folder, 'marginal_smooth'))
    plt.close('all')


def plot_marginal_pdf(path, C_limits):

    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']
    N_params = len(C_limits)
    max_value = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i)+str(j)] = np.loadtxt(os.path.join(path['output'], 'marginal{}{}'.format(i, j)))
                max_value = max(max_value, np.max(data[str(i)+str(j)]))
    max_value = int(max_value)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    # cmaplist[0] = 'black'   # force the first color entry to be black
    cmaplist[0] = 'white' # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    confidence = np.loadtxt(os.path.join(path['output'], 'confidence'))
    fig = plt.figure(figsize=(1.25*fig_width, 1.1*fig_width))
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(path['output'], 'marginal_{}'.format(i)))

                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)

                if i == 0:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                if i == 1:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                if i == 2:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
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
    # cax = plt.axes([0.05, 0.1, 0.01, 0.26])
    # plt.colorbar(im, cax=cax)   #, ticks=np.arange(max_value+1))

    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3, hspace=0.1, bottom=0.1, top=0.98)

    fig.savefig(os.path.join(path['plots'], 'marginal'))
    plt.close('all')


def plot_dist_pdf(path, dist, x):

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.hist(dist, bins=100, alpha=0.8)
    eps = np.percentile(dist, q=int(x * 100))
    print('eps =', eps)
    ax.axvline(eps, label='eps')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'pdf($\rho$)')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(path['plots'], 'dist'))
    plt.close('all')
    return eps


def plot_bootstrapping_pdf(path, dist, x, i, C_limit, C_final):

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.hist(dist, bins=20, alpha=0.8, range=C_limit)
    q1 = np.percentile(dist, q=x)
    ax.axvline(q1, color='g', label=r'$2.5\%$')
    q2 = np.percentile(dist, q=100-x)
    ax.axvline(q2, color='g', label=r'$97.5\%$')
    ax.axvline(C_final, color='r', label=r'$C_{max}$')
    ax.set_xlabel(r'$C_{}$'.format(i))
    ax.set_ylabel(r'pdf')
    plt.legend()
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(path['plots'], 'bootstrapping'+str(i)))
    plt.close('all')
    return q1, q2


def plot_prior(params_names, C_limits, data_folder, plot_folder):
    N_params = len(params_names)
    max_value = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i) + str(j)] = np.loadtxt(os.path.join(data_folder, 'marginal_smooth{}{}'.format(i, j)))
                max_value = max(max_value, np.max(data[str(i) + str(j)]))
    max_value = int(max_value)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    # cmaplist[0] = 'black'   # force the first color entry to be black
    cmaplist[0] = 'white'  # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    # confidence = np.loadtxt(os.path.join(path['output'], 'confidence'))
    fig = plt.figure(figsize=(1.25 * fig_width, 1.1 * fig_width))
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(data_folder, 'marginal_smooth{}'.format(i)))

                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                c_final_smooth = np.loadtxt(os.path.join(os.path.join(data_folder, 'C_final_smooth')))
                # ax.axvline(confidence[i, 0], linestyle='--', color='b', label=r'$90\%$ interval')
                # ax.axvline(confidence[i, 1], linestyle='--', color='b')
                if len(c_final_smooth.shape) == 1:
                    ax.axvline(c_final_smooth[i], linestyle='--', color='r', label='MAP smooth')
                elif len(c_final_smooth) < 4:
                    for C in c_final_smooth:
                        ax.axvline(C[i], linestyle='--', color='b', label='MAP smooth')

                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 15))
                ax.set_xlabel(params_names[i], labelpad=2)
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

                # ax.yaxis.set_major_formatter(plt.NullFormatter())
                # ax.yaxis.set_major_locator(plt.NullLocator())
                # ax.tick_params(axis='both', which='minor', direction='in')
                # ax.tick_params(axis='both', which='major', pad=0.8)

                if i == 0:
                    if N_params == 3:
                        ax.legend(bbox_to_anchor=(2.35, -1.5), fancybox=True)
                    elif N_params == 4:
                        ax.legend(bbox_to_anchor=(3, -2.75), fancybox=True)
                    textstr = '\n'.join((
                        r'$C_1=%.3f$' % (c_final_smooth[0],),
                        r'$C_2=%.3f$' % (c_final_smooth[1],),
                        r'$C_{\epsilon1}=%.3f$' % (c_final_smooth[2],),
                        r'$C_{\epsilon2}=%.3f$' % (c_final_smooth[3],)))
                    ax.text(0.15, -1.6, textstr, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', linespacing=1.5)
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])

                # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                # ax.xaxis.set_major_formatter(plt.NullFormatter())
                # ax.yaxis.set_tick_params(direction='in')

                # ax.tick_params(axis='both', which='minor', direction='in')
                # ax.tick_params(axis='both', which='major', pad=0.8)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 15))
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])
                im = ax.imshow(data[str(i) + str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
    # cax = plt.axes([0.05, 0.1, 0.01, 0.26])
    # plt.colorbar(im, cax=cax)   #, ticks=np.arange(max_value+1))
    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3, hspace=0.1, bottom=0.1, top=0.98)
    fig.savefig(os.path.join(plot_folder, 'marginal_smooth'))
    plt.close('all')


def main():
    basefolder = './'

    path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots/')}
    path['calibration'] = os.path.join(path['output'], 'calibration/')
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])

    C_limits = np.loadtxt(os.path.join(path['calibration'], 'C_limits'))
    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']

    plot_prior(params_names, C_limits, path['calibration'], path['plots'])
    # plot_marginal_smooth_pdf(path, C_limits)
    # s0 = 3.3
    # beta = [0.125, 0.25, 0.5, 0.75, 1]
    # ####################################################################################################################
    # plot_marginal_pdf(path, C_limits)
    # plot_marginal_smooth_pdf(path, C_limits)
    #
    # c = np.loadtxt(os.path.join(path['output'], 'C_final_smooth'))
    # print('C_final_smooth: ', c)
    # err = np.zeros(13)


if __name__ == '__main__':
    main()