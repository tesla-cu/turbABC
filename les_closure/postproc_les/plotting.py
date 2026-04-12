import gc
import glob

import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import string

# plt.style.use('dark_background')
# mpl.style.use(['dark_background','mystyle'])
# mpl.style.use(['mystyle'])

single_column = 255
oneandhalf_column = 397
double_column = 539
fig_width_pt = double_column
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 10
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


def imagesc(Arrays, map_bounds, titles, name=None):

    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
    norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

    axis = [0, 2 * np.pi, 0, 2 * np.pi]
    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(fig_width, 0.5*fig_width))
        # plt.axes([0.125, 0.2, 0.95 - 0.125, 0.95 - 0.2])
        k = 0
        for ax in axes.flat:
            im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
            ax.set_title(titles[k])
            ax.set_adjustable('box-forced')
            ax.set_xlabel(r'$x$')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            k += 1
        axes[0].set_ylabel(r'$y$')
        # cbar_ax = fig.add_axes([0.89, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
        cbar_ax = fig.add_axes([0.83, 0.21, 0.017, 0.67])
        fig.subplots_adjust(left=0.11, right=0.8, wspace=0.1, bottom=0.2, top=0.9)
        fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
    else:
        fig = plt.figure(figsize=(6.5, 5))
        ax = plt.gca()
        im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest")
        plt.colorbar(im, fraction=0.05, pad=0.04)
    if name:
        # pickle.dump(ax, open(self.folder + name, 'wb'))
        fig.savefig(name)
    plt.close('all')

    # def histogram(field, bins, pdf=None, label=None, log=False):
    #     plt.figure(figsize=(6, 4))
    #     plt.hist(field, bins=bins, alpha=0.4, normed=1)
    #
    #     # h, edges = np.histogram(field, bins=bins, range=[-2,2], normed=1)
    #     if pdf:
    #         x = np.linspace(min(field), max(field), 100)
    #         mu = np.mean(field)
    #         sigma = np.std(field)
    #         plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=3, label='Gaussian')
    #         plt.legend(loc=0)
    #     if log:
    #         plt.yscale('log', nonposy='clip')
    #     if label:
    #         plt.xlabel(label)
    #         plt.ylabel('pdf(' + label + ')')
    #     plt.axis(xmin=np.min(field), xmax=np.max(field))  # xmax=4xmax = np.max(field)
    #     plt.show()
    #     gc.collect()

########################################################################################################################
# Plot initial data info
########################################################################################################################


def compare_filter_fields(hit_data, les_data, test_data, map_bounds, folder):

    name = os.path.join(folder, 'compare_velocity')
    if test_data:
        imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127], test_data['v'][:, :, 127]],
                map_bounds=map_bounds, name=name,
                titles=[r'$v$', r'$\widetilde{v}$', r'$\widehat{\widetilde{v}}$'])
    else:
        imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127]],
                map_bounds=map_bounds, name=name,
                titles=[r'$v$', r'$\widetilde{v}$'])


def vel_fields(data, scale, map_bounds, folder):
    """ Plot velocity fields"""
    if scale == 'DNS':
        titles = [r'$u$', r'$v$', r'$w$']
        name = os.path.join(folder, 'DNS_velocities')
    elif scale == 'LES':
        titles = [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$']
        name = os.path.join(folder, 'LES_velocities')
    elif scale == 'TEST':
        titles = [r'$\widehat{\widetilde{u}}$', r'$\widehat{\widetilde{v}}$', r'$\widehat{\widetilde{w}}$']
        name = os.path.join(folder, 'TEST_velocities')
    imagesc([data['u'][:, :, 127], data['v'][:, :, 127], data['w'][:, :, 127]],
            name=name, titles=titles, map_bounds=map_bounds)


def sigma_field(tau, scale, folder):

    map_bounds = np.linspace(-0.2, 0.2, 9)
    if scale == 'LES':
        name = os.path.join(folder, 'sigma_LES')
        titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

    elif scale == 'TEST':
        name = os.path.join(folder, 'sigma_TEST')
        titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']

    imagesc([tau['uu'][:, :, 127], tau['uv'][:, :, 127], tau['uw'][:, :, 127]],
            map_bounds, name=name, titles=titles)


def sum_stat(les, test, bins, domain, folder, name):

    labels = ['LES', 'test']

    if name == 'sigma_pdf_log':
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        if test:
            titles = [r'$\sigma_{11},\ \widehat{\sigma}_{11}$',
                    r'$\sigma_{12},\ \widehat{\sigma}_{12}$',
                    r'$\sigma_{13},\ \widehat{\sigma}_{13}$']
        else:
            titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = les.sum_stat_true[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, bins, domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label=labels[0])
            axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        if test:
            for ind, i in enumerate(['uu', 'uv', 'uw']):
                data = test.sum_stat_true[i].flatten()
                x, y = utils.pdf_from_array_with_x(data, bins, domain)
                axarr[ind].plot(x, y, 'g', linewidth=2, label=labels[1])
                axarr[ind].set_xlabel(titles[ind])

        axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)

    elif name == 'production_pdf_log':
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        if test:
            title = r'$\widehat{\widetilde{P}}'
        else:
            title = r'$\widetilde{P}'

        data = les.sum_stat_true.flatten()
        x, y = utils.pdf_from_array_with_x(data, bins, domain)
        ax.plot(x, y, 'r', linewidth=2, label=labels[0])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        if test:
            data = test.sum_stat_true.flatten()
            x, y = utils.pdf_from_array_with_x(data, bins, domain)
            ax.plot(x, y, 'g', linewidth=2, label=labels[1])
            ax.set_xlabel(title)

        ax.axis(xmin=-5, xmax=5, ymin=1e-5)
        ax.set_ylabel('pdf')
        ax.set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(os.path.join(folder, name))
    plt.close('all')


def S_pdf(folder):

    name = 'S_pdf_' + str(g.TEST_sp.M)
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.5))
    titles = [r'$\widetilde{S}_{11}/\widehat{\widetilde{S}}_{11}$',
              r'$\widetilde{S}_{12}/\widehat{\widetilde{S}}_{12}$',
              r'$\widetilde{S}_{13}/\widehat{\widetilde{S}}_{13}$']
    strain = g.LES.S
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = strain[i].flatten()
        x, y = utils.pdf_from_array_with_x(data, g.bins, [-2, 2])
        axarr[ind].plot(x, y, 'r', linewidth=1, label='LES')
        axarr[ind].set_xlabel(titles[ind])
    strain = g.TEST.S
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = strain[i].flatten()
        x, y = utils.pdf_from_array_with_x(data, g.bins, [-2, 2])
        axarr[ind].plot(x, y, 'g', linewidth=1, label='test')
    strain = g.TEST_sp.S
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = strain[i].flatten()
        x, y = utils.pdf_from_array_with_x(data, g.bins, [-2, 2])
        axarr[ind].plot(x, y, 'b', linewidth=1, label='test sparse M='+str(g.TEST_sp.M))

    # axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    axarr[0].set_ylabel('pdf')
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)
    fig.savefig(folder + name)
    del fig, axarr
    gc.collect()



def spectra(folder):

    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    files = glob.glob(folder + '*.spectra')

    labels = ['DNS', 'LES', 'test']

    for k in range(len(files)):
        f = open(files[k], 'r')
        data = np.array(f.readlines()).astype(np.float)
        x = np.arange(len(data))
        ax.loglog(x, data, '-', linewidth=2, label=labels[k])

    y = 7.2e14 * np.power(x, -5 / 3)
    ax.loglog(x, y, 'r--', label=r'$-5/3$ slope')
    ax.set_title('Spectra')
    ax.set_ylabel(r'$E$')
    ax.set_xlabel(r'k')
    ax.axis(ymin=1e6)
    plt.legend(loc=0)

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(folder + 'spectra')

def tau_abc(Cs_abc):
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(18, 6))
    titles = [r'$\widehat{T}_{11}$', r'$\widehat{T}_{12}$', r'$\widehat{T}_{13}$']

    x = np.linspace(g.domain[0], g.domain[1], g.bins)
    plots = []
    labels = []
    for ind, i in enumerate(['uu', 'uv', 'uv']):
        line = axarr[ind].plot(x, g.TEST.tau_pdf_true[i], linewidth=3, label='true pdf')
        if ind == 0:
            plots.append(line)
            labels.append('true pdf')
        axarr[ind].set_xlabel(titles[ind])
    for C_s in Cs_abc:
        tau = g.TEST.Reynolds_stresses_from_Cs(C_s)
        for ind, i in enumerate(['uu', 'uv', 'uv']):
            x, y = utils.pdf_from_array(tau[i].flatten(), g.bins, g.domain)
            line = axarr[ind].plot(x, y, linewidth=1, label=r'$C_s \approx\  $' + str(round(C_s, 3)))
            if ind == 0:
                plots.append(line)
                labels.append(r'$C_s \approx $' + str(round(C_s, 3)))
    axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    axarr[0].set_ylabel('pdf')
    axarr[0].set_yscale('log', nonposy='clip')
    # Shrink current axis's width
    for i in range(3):
        box = axarr[i].get_position()
        axarr[i].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(1.6, 1.1), fancybox=True, shadow=True, ncol=2)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('tau_abc.eps')
    del fig, axarr
    gc.collect()


def plot_dist_pdf(dist, x, folder):

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.hist(dist, bins=100, alpha=0.8)
    eps = np.percentile(dist, q=int(x * 100))
    print('eps =', eps)
    ax.axvline(eps, label='eps')
    # ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'pdf($\rho$)')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(folder, 'dist'))
    plt.close('all')


########################################################################################################################
#
########################################################################################################################
params_names = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']


def plot_marginal_pdf(N_params, output, plot_folder, C_limits, name=''):

    max_value = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i)+str(j)] = np.loadtxt(os.path.join(output, 'marginal' + name + str(i) + str(j)))
                max_value = max(max_value, np.max(data[str(i)+str(j)]))
    max_value = int(max_value)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    # cmaplist[0] = 'black' #'white' # force the first color entry to be white
    cmaplist[0] = 'white' # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    fig = plt.figure(figsize=(1.25*fig_width, 1.1*fig_width))
    if N_params == 6:
        fig = plt.figure(figsize=(2 * fig_width, 1.9 * fig_width))
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(output, 'marginal' + name + str(i)))
                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                # c_final_dist = np.loadtxt(os.path.join(output, 'C_final_dist'))
                # ax.axvline(mean, linestyle='--', color='g', label='mean')
                # ax.axvline(max, linestyle='--', color='r', label='max')
                # ax.axvline(c_final_dist[i], linestyle='--', color='g', label='min distance')
                if name == '':
                    c_final_joint = np.loadtxt(os.path.join(output, 'C_final_joint'))
                    if len(c_final_joint.shape) == 1:
                        ax.axvline(c_final_joint[i], linestyle='--', color='b', label='max of joint pdf')
                    elif len(c_final_joint) < 4:
                        for C in c_final_joint:
                            ax.axvline(C[i], linestyle='--', color='b', label='joint max')
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
                # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                if i == 2:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                if i == 0:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_locator(plt.NullLocator())
                ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)
                ax.set_xlabel(params_names[i], labelpad=2)
                if i == 0:
                    if N_params == 3:
                        ax.legend(bbox_to_anchor=(2.35, -1.5), fancybox=True)
                    elif N_params == 4:
                        ax.legend(bbox_to_anchor=(3, -2.75), fancybox=True)
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                edges = np.loadtxt(os.path.join(output, 'marginal_bins' + name + str(i)+str(j)))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                # ax.yaxis.tick_right()
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.yaxis.set_tick_params(direction='in')
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                if (j == 2 or j == 3) and i == 1:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                # if j != (N_params-1):
                #     ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.tick_params(axis='both', which='minor', direction='in')
                ax.tick_params(axis='both', which='major', pad=0.8)
                ext = (edges[0, 0], edges[0, -1], edges[1, 0], edges[1, -1])

                im = ax.imshow(data[str(i)+str(j)].T, origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
    cax = plt.axes([0.05, 0.1, 0.01, 0.26])
    plt.colorbar(im, cax=cax) #, ticks=np.arange(max_value+1))

    if N_params == 3:
        # fig.subplots_adjust(left=0.02, right=0.9, wspace=0.1, hspace=0.1, bottom=0.1, top=0.98)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.28, hspace=0.1, bottom=0.1, top=0.98)
    elif N_params == 4:
        fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3, hspace=0.1, bottom=0.1, top=0.98)
    elif N_params == 6:
        fig.subplots_adjust(left=0.05, right=0.98, wspace=0.45, hspace=0.35, bottom=0.08, top=0.98)

    fig.savefig(os.path.join(plot_folder, 'marginal' + name))
    plt.close('all')


def plot_marginal_smooth_pdf_3(N_params, output, plot_folder, C_limits, name=''):

    max_value1 = 0.0
    max_value2 = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i)+str(j)] = np.loadtxt(os.path.join(output, 'marginal_smooth' + name + str(i) + str(j)))
                norm = np.sum(data[str(i)+str(j)])
                print('norm = ', norm, np.max(data[str(i)+str(j)]))
                data[str(i) + str(j)] /= norm
                max_value1 = max(max_value1, np.max(data[str(i)+str(j)]))
                print(np.max(data[str(i) + str(j)]))
            if i > j:
                data[str(i) + str(j)] = np.loadtxt(os.path.join(output, 'conditional_smooth' + name + str(i) + str(j)))
                norm = np.sum(data[str(i)+str(j)])
                print('norm = ', norm, np.max(data[str(i)+str(j)]))
                data[str(i) + str(j)] /= norm
                max_value2 = max(max_value2, np.max(data[str(i)+str(j)]))
                print('this', np.max(data[str(i) + str(j)]))

    print(max_value1, max_value2)
    ###################################################################################################################
    cmap2 = cm.inferno  # define the colormap
    cmap = plt.cm.BuPu  # define the colormap
    cmaplist = [cmap(i) for i in reversed(range(cmap.N))]  # extract all colors from the map
    cmaplist2 = [cmap2(i) for i in (range(cmap2.N))]  # extract all colors from the map
    gamma1 = 1
    gamma2 = 1
    ###################################################################################################################
    # cmap2 = plt.cm.Greys   # define the colormap
    # cmap = plt.cm.Greys  # define the colormap
    # cmaplist = [cmap(i) for i in (range(cmap.N))]  # extract all colors from the map
    # cmaplist2 = [cmap2(i) for i in (range(cmap2.N))]  # extract all colors from the map
    # gamma1 = 1
    # gamma2 = 1
    cmaplist[0] = 'black' #'white' # force the first color entry to be white
    cmaplist2[0] = 'black'  # force the first color entry to be white

    # cmaplist[0] = 'white' # force the first color entry to be white
    # cmaplist2[0] = 'white'  # force the first color entry to be white
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist)
    cmap2 = LinearSegmentedColormap.from_list('Custom cmap', cmaplist2)

    fig = plt.figure(figsize=(fig_width, 0.6*fig_width))
    # ticks = [0.05, 0.1, 0.1]   # sigma
    ticks = [0.03, 0.2, 0.2]   # production and both
    ax_diag = dict()
    length = np.empty(N_params)
    for i in range(N_params):
        length[i] = C_limits[i, 1] - C_limits[i, 0]
    print(length)
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(output, 'marginal_smooth' + name + str(i)))
                c_final_smooth = np.loadtxt(os.path.join(output, 'C_final_smooth'))
                ax_diag[i] = plt.subplot2grid((N_params, N_params), (i, i))
                ax_diag[i].plot(data_marg[0], data_marg[1], 'w', label='marginal pdf')
                labelsrt = 'maximum of \n joint pdf:'
                if len(c_final_smooth.shape) == 1:
                    ax_diag[i].axvline(c_final_smooth[i], linestyle='--', color='b', label=labelsrt)
                elif len(c_final_smooth) < 4:
                    for C in c_final_smooth:
                        ax_diag.axvline(C[i], linestyle='--', color='b', label='joint max')
                ax_diag[i].axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
                ax_diag[i].yaxis.set_major_formatter(plt.NullFormatter())
                ax_diag[i].yaxis.set_major_locator(plt.NullLocator())
                ax_diag[i].tick_params(axis='both', which='major', pad=2)
                # ax.set_ylabel('pdf', labelpad=3)
                ax_diag[i].xaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                # if True and i == 1:
                #     ax_diag[i].text(0.03, 0.09, '('+string.ascii_lowercase[i*N_params+j]+')',
                #                 transform=ax_diag[i].transAxes, size=10, weight='black')
                # else:
                #     ax_diag[i].text(0.03, 0.85, '('+string.ascii_lowercase[i*N_params+j]+')',
                #                 transform=ax_diag[i].transAxes, size=10, weight='black')

                if i != (N_params - 1):
                    ax_diag[i].xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax_diag[i].set_xlabel(params_names[i], labelpad=2)

                # if i == 0:
                #     ax.legend(bbox_to_anchor=(5.25, 1.07), fancybox=True)
                #     textstr = '\n'.join((r'$C_1=%.3f$' % (c_final_smooth[0],),
                #                          r'$C_2=%.3f$' % (c_final_smooth[1],),
                #                          r'$C_3=%.3f$' % (c_final_smooth[2],)))
                #     ax.text(4.3, 0.35, textstr, transform=ax.transAxes, verticalalignment='top', linespacing=1.5)
                #     ax.text(4.4, -0.35, '2D marginal', transform=ax.transAxes, horizontalalignment='center',
                #             rotation=90, linespacing=1.5)
                #     ax.text(4.4, -1.48, 'conditional', transform=ax.transAxes, horizontalalignment='center',
                #             rotation=90, linespacing=1.5)
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])

                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                ax.tick_params(axis='both', which='major', pad=1.5)
                ax.yaxis.tick_right()

                # ax.text(0.03, 0.85, '('+string.ascii_lowercase[i*N_params+j]+')',
                #         transform=ax.transAxes, size=10, weight='black')

                if j != N_params-1:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_ylabel(params_names[i])
                    ax.yaxis.set_label_position("right")
                # if i != N_params-1 and j == N_params-1:
                #     if i == 0 and j==1:
                #         pass
                #     else:
                #         ax.yaxis.set_major_formatter(plt.NullFormatter())
                # else:
                #     ax.set_ylabel(params_names[i], labelpad=2)
                # if i == 0 and j == 1:
                #     ax.set_ylabel(params_names[i], labelpad=2)

                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])
                im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, norm=colors.PowerNorm(gamma=gamma1), vmin=0, vmax=max_value1)
            elif i > j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])
                ax.tick_params(axis='both', which='major', pad=2)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                # ax.text(0.03, 0.85, '('+string.ascii_lowercase[i*N_params+j]+')',
                #         transform=ax.transAxes, size=10, weight='black')
                if j == 0:
                    ax.set_ylabel(params_names[i])
                else:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                if i != (N_params - 1):
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_xlabel(params_names[j])

                im_cond = ax.imshow(data[str(i) + str(j)].T, origin='lower', cmap=cmap2,
                                    aspect='auto', extent=ext, norm=colors.PowerNorm(gamma=gamma2),
                                    vmin=0, vmax=max_value2)
                ax.axvline(c_final_smooth[j], linestyle='--', color='b')
                ax.axhline(c_final_smooth[i], linestyle='--', color='b')
    # sigma
    # ticks1 = np.arange(1, 9)/10000
    # ticks2 = np.arange(3, 9)/10000
    # prod
    # ticks1 = np.arange(2, 20, 3)/10000
    # ticks2 = np.arange(2, 40, 5)/10000
    # both
    ticks1 = np.arange(1, 20, 2)/10000
    ticks2 = np.arange(1, 40, 2)/10000

    cax = plt.axes([0.18, 0.705, 0.01, 0.275])
    plt.colorbar(im, cax=cax, format=ticker.FormatStrFormatter('%.0e'), ticks=ticks1)
    cax.yaxis.set_ticks_position('left')
    cax = plt.axes([0.81, 0.1, 0.01, 0.275])
    plt.colorbar(im_cond, cax=cax, format=ticker.FormatStrFormatter('%.1e'), ticks=ticks2)

    fig.subplots_adjust(left=0.2, right=0.8, wspace=0.1, hspace=0.1, bottom=0.1, top=0.98)
    fig.savefig(os.path.join(plot_folder, 'marginal_smooth' + name))
    plt.close('all')


# def plot_3d(path):
#     Z = np.load(os.path.join(path['output'], 'Z.npz'))['Z']
#     print(Z.shape)
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')


def plot_marginal_smooth_pdf_4(N_params, output, plot_folder, C_limits, name=''):

    max_value1, max_value2 = 0.0, 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            for i in range(N_params):
                for j in range(N_params):
                    if i < j:
                        data[str(i) + str(j)] = np.loadtxt(
                            os.path.join(output, 'marginal_smooth' + name + str(i) + str(j)))
                        norm = np.sum(data[str(i) + str(j)])
                        print('norm = ', norm, np.max(data[str(i) + str(j)]))
                        data[str(i) + str(j)] /= norm
                        max_value1 = max(max_value1, np.max(data[str(i) + str(j)]))
                        print(np.max(data[str(i) + str(j)]))
                    if i > j:
                        data[str(i) + str(j)] = np.loadtxt(
                            os.path.join(output, 'conditional_smooth' + name + str(i) + str(j)))
                        norm = np.sum(data[str(i) + str(j)])
                        print('norm = ', norm, np.max(data[str(i) + str(j)]))
                        data[str(i) + str(j)] /= norm
                        max_value2 = max(max_value2, np.max(data[str(i) + str(j)]))
                    print(max_value1, max_value2)
            # if i < j:
            #     data[str(i)+str(j)] = np.loadtxt(os.path.join(output, 'marginal_smooth' + name + str(i) + str(j)))
            #     max_value = max(max_value, np.max(data[str(i)+str(j)]))
            # if i > j:
            #     data[str(i) + str(j)] = np.loadtxt(os.path.join(output, 'conditional_smooth' + name + str(i) + str(j)))
    ###################################################################################################
    # cmap = plt.cm.jet  # define the colormap
    # cmap2 = plt.cm.gist_stern  # define the colormap
    # cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    # cmaplist2 = [cmap2(i) for i in reversed(range(cmap2.N))]  # extract all colors from the .jet map
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
    cmaplist[0] = 'black' #'white' # force the first color entry to be white
    cmaplist2[0] = 'black'  # 'white' # force the first color entry to be white
    # cmaplist[0] = 'white' # force the first color entry to be white
    # cmaplist2[0] = 'white'  # force the first color entry to be white
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist)
    cmap2 = LinearSegmentedColormap.from_list('Custom cmap', cmaplist2)

    fig = plt.figure(figsize=(fig_width, 0.6*fig_width))

    # ticks = [0.03, 0.3, 0.2, 0.3]   # production
    # ticks = [0.04, 0.1, 0.1, 0.1]  # sigma
    ticks = [0.03, 0.2, 0.2, 0.3]  # both
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(output, 'marginal_smooth' + name + str(i)))
                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1], 'w', label='marginal pdf')

                c_final_smooth = np.loadtxt(os.path.join(output, 'C_final_smooth'))
                labelsrt = 'maximum of \n joint pdf'
                if len(c_final_smooth.shape) == 1:
                    ax.axvline(c_final_smooth[i], linestyle='--', color='b', label=labelsrt)
                elif len(c_final_smooth) < 4:
                    for C in c_final_smooth:
                        ax.axvline(C[i], linestyle='--', color='b', label='joint max')
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_locator(plt.NullLocator())
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                ax.tick_params(axis='both', which='major', pad=2)
                # if True and (j == 1 or j == 0):
                #     ax.text(0.02, 0.07, '(' + string.ascii_lowercase[i * N_params + j] + ')',
                #             transform=ax.transAxes, size=10, weight='black')
                # else:
                #     ax.text(0.02, 0.85, '('+string.ascii_lowercase[i*N_params+j]+')',
                #         transform=ax.transAxes, size=10, weight='black')

                if i != (N_params - 1):
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_xlabel(params_names[i], labelpad=2)

                # if i == 0:
                #     ax.legend(bbox_to_anchor=(7.15, 1.08), fancybox=True)
                #     textstr = '\n'.join((
                #         r'$C_1=%.3f$' % (c_final_smooth[0],),
                #         r'$C_2=%.3f$' % (c_final_smooth[1],),
                #         r'$C_3=%.3f$' % (c_final_smooth[2],),
                #         r'$C_4=%.3f$' % (c_final_smooth[3],)))
                #     ax.text(5.8, -0.1, textstr, transform=ax.transAxes, verticalalignment='top', linespacing=1.5)
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])

                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                ax.tick_params(axis='both', which='major', pad=1.5)
                # if True and i != 2:
                #     ax.text(0.02, 0.07, '(' + string.ascii_lowercase[i * N_params + j] + ')',
                #             transform=ax.transAxes, size=10, weight='black')
                # else:
                #     ax.text(0.02, 0.85, '('+string.ascii_lowercase[i*N_params+j]+')',
                #         transform=ax.transAxes, size=10, weight='black')
                ax.yaxis.tick_right()
                # if i == 0:
                #     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

                if j != N_params-1:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_ylabel(params_names[i])
                    ax.yaxis.set_label_position("right")


                # if i != N_params-1 and j == N_params-1:
                #     if i == 0 and j==1:
                #         pass
                #     else:
                #         ax.yaxis.set_major_formatter(plt.NullFormatter())
                # else:
                #     ax.set_ylabel(params_names[i], labelpad=2)
                # if i == 0 and j == 1:
                #     ax.set_ylabel(params_names[i], labelpad=2)
                # if i != (N_params-1) and j == (N_params - 1):
                #     ax.set_ylabel(params_names[i], labelpad=1)
                #     ax.yaxis.tick_right()
                #     ax.yaxis.set_label_position("right")

                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])
                im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, norm=colors.PowerNorm(gamma=gamma1), vmin=0, vmax=max_value1)
            elif i > j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                ext = (C_limits[j, 0], C_limits[j, 1], C_limits[i, 0], C_limits[i, 1])
                ax.tick_params(axis='both', which='major', pad=2)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
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

                im_cond = ax.imshow(data[str(i) + str(j)].T, origin='lower', cmap=cmap2, aspect='auto', extent=ext,
                                    norm=colors.PowerNorm(gamma=gamma2), vmax=max_value2)
                ax.axvline(c_final_smooth[j], linestyle='--', color='b')
                ax.axhline(c_final_smooth[i], linestyle='--', color='b')

    # sigma
    ticks1 = np.arange(1, 20, 2)/1000
    ticks2 = np.arange(1, 40, 2)/1000
    cax = plt.axes([0.18, 0.77, 0.01, 0.21])
    cbar = plt.colorbar(im, cax=cax, format=ticker.FormatStrFormatter('%.0e'), ticks=ticks1 )
    cax.yaxis.set_ticks_position('left')
    # cbar.ax.get_yaxis().labelpad = -49
    # cbar.ax.set_ylabel('2D marginal', fontsize=10)
    # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    cax = plt.axes([0.81, 0.08, 0.01, 0.21])
    cbar1 = plt.colorbar(im_cond, cax=cax, format=ticker.FormatStrFormatter('%.0e'), ticks=ticks2)
    # cbar1.ax.get_yaxis().labelpad = -40
    # cbar1.ax.set_ylabel('conditional', fontsize=10)

    # horizontal
    # cax = plt.axes([0.78, 0.425, 0.19, 0.015])
    # cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    # cbar.ax.set_title('2D marginal', fontsize=10)
    # cax = plt.axes([0.78, 0.2, 0.19, 0.015])
    # cbar1 = plt.colorbar(im_cond, cax=cax, orientation='horizontal')
    # cbar1.ax.set_title('conditional', fontsize=10)

    fig.subplots_adjust(left=0.2, right=0.8, wspace=0.1, hspace=0.1, bottom=0.08, top=0.98)
    fig.savefig(os.path.join(plot_folder, 'marginal_smooth' + name))
    plt.close('all')


def plot_scatter(N_params, C_limits, visua, accepted, dist):

    for i in range(N_params):
        x = accepted[:, i]
        fig = plt.figure(figsize=(0.75*fig_width,0.5*fig_width))
        ax = plt.axes()
        ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymax=1.1*np.max(dist), ymin=0)
        ax.scatter(x, dist, marker=".", color='blue')
        ax.set_xlabel(params_names[i])
        ax.set_ylabel(r'$\sum_{i,j}\rho(\mathcal{S}_{ij}^{\mathcal{F}},\mathcal{S}_{ij})$')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        fig.subplots_adjust(left=0.245, right=0.96, bottom=0.21, top=0.97)
        fig.savefig(os.path.join(visua, 'scatter_plot_'+params_names[i][1:-1]))
    plt.close('all')


def plot_compare_tau(visua, output, sum_stat, scale):


    fig, axarr = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(fig_width, 0.35*fig_width))
    titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$', r'$\sigma_{ij}S_{ij}$']
    y_true = dict()
    y_min_dist = dict()
    y_max_joint = dict()
    x = np.loadtxt(os.path.join(output, 'sum_stat_bins'))[0]
    for ind, key in enumerate(['uu', 'uv', 'uw']):
        # Plot true pdf
        y_true[key] = np.loadtxt(os.path.join(output, 'sum_stat_true'))[ind]
        axarr[ind].plot(x, y_true[key], 'r', linewidth=1, label='true')
        # plot min dist pdf
        # y_min_dist[key] = np.loadtxt(os.path.join(output, 'sum_stat_min_dist_' + scale))[ind]
        # axarr[ind].plot(x, y_min_dist[key], 'g', linewidth=1, label='modeled dist')
        # # plot max joint
        # if os.path.isfile(os.path.join(output, 'sum_stat_max_joint_' + scale)):
        #     y_max_joint[key] = np.loadtxt(os.path.join(output, 'sum_stat_max_joint_' + scale))[ind]
        #     axarr[ind].plot(x, y_max_joint[key], 'b', linewidth=1, label='modeled')
        # plot max smooth
        if os.path.isfile(os.path.join(output, 'sum_stat_max_smooth_' + scale)):
            y_max_joint[key] = np.loadtxt(os.path.join(output, 'sum_stat_max_smooth_' + scale))[ind]
            print(x, y_max_joint[key])
            axarr[ind].plot(x, y_max_joint[key], 'b', linewidth=1, label='modeled')
        # # plot max marginal
        # x, y = utils.pdf_from_array_with_x(tau_modeled_marginal[key].flatten(), g.bins, g.domain)
        # axarr[ind].plot(x, y, 'm', linewidth=2, label='modeled marginal max')
        axarr[ind].set_xlabel(titles[ind], labelpad=2)
        axarr[ind].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axarr[0].set_ylabel('ln(pdf)', labelpad=2)
    axarr[0].yaxis.set_major_locator(ticker.MultipleLocator(2))
    fig.subplots_adjust(left=0.12, right=0.98, wspace=0.1, bottom=0.22, top=0.80)

    x = np.loadtxt(os.path.join(output, 'sum_stat_bins'))[1]
    y_true = np.loadtxt(os.path.join(output, 'sum_stat_true'))[3]
    axarr[3].plot(x, y_true, 'r', linewidth=1, label='true')
    if os.path.isfile(os.path.join(output, 'sum_stat_max_smooth_' + scale)):
        y_max_joint = np.loadtxt(os.path.join(output, 'sum_stat_max_smooth_' + scale))[3]
        axarr[3].plot(x, y_max_joint, 'b', linewidth=1, label='modeled')
    # y_min_dist = np.loadtxt(os.path.join(output, 'sum_stat_min_dist_' + scale))[3]
    # axarr[3].plot(x, y_min_dist, 'g', linewidth=1)
    axarr[3].set_xlabel(titles[3], labelpad=2)
    axarr[3].set_ylim(bottom=-6)
    axarr[3].xaxis.set_major_locator(ticker.MultipleLocator(2))

    axarr[1].legend(loc='upper center', bbox_to_anchor=(0.3, 1.35), fancybox=False, shadow=False, ncol=3)

    fig.savefig(os.path.join(visua, 'compare_sum_stat_' + scale))
    plt.close('all')

def plot_sum_stat(path, name):

    sum_stat = np.loadtxt(os.path.join(path['output'], 'sum_stat_true'))
    bins = np.loadtxt(os.path.join(path['output'], 'sum_stat_bins'))

    if name == 'sigma_pdf_log':
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

        for i in range(len(sum_stat)):
            axarr[i].plot(bins, sum_stat[i], 'r', linewidth=2)
            axarr[i].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(0.45))
            axarr[i].set_xlabel(titles[i])

        axarr[0].axis(xmin=-0.5, xmax=0.5)
        axarr[0].set_ylabel('ln(pdf)')
        # axarr[0].set_yscale('log')
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)

    elif name == 'production_pdf_log':
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        title = r'$\widetilde{P}$'

        ax.plot(bins, sum_stat, 'r', linewidth=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.axis(xmin=-5, xmax=5)
        ax.set_xlabel(title)
        ax.set_ylabel('ln(pdf)')
        # ax.set_yscale('log')
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

    fig.savefig(os.path.join(path['visua'], name))
    plt.close('all')