import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import abc_alg as m
import utils

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
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']-1
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
folder_data = './valid_data/'
axi_exp_k = np.loadtxt(os.path.join(folder_data, 'axi_exp_k.txt'))
axi_exp_b = np.loadtxt(os.path.join(folder_data, 'axi_exp_b.txt'))
axi_con_k = np.loadtxt(os.path.join(folder_data, 'axi_con_k.txt'))
axi_con_b = np.loadtxt(os.path.join(folder_data, 'axi_con_b.txt'))
shear_k = np.loadtxt(os.path.join(folder_data, 'shear_k.txt'))
plane_k = np.loadtxt(os.path.join(folder_data, 'plane_k.txt'))
plane_b = np.loadtxt(os.path.join(folder_data, 'plane_b.txt'))
period1_k = np.loadtxt(os.path.join(folder_data, 'period1_k.txt'))
period2_k = np.loadtxt(os.path.join(folder_data, 'period2_k.txt'))
period3_k = np.loadtxt(os.path.join(folder_data, 'period3_k.txt'))
period4_k = np.loadtxt(os.path.join(folder_data, 'period4_k.txt'))
period5_k = np.loadtxt(os.path.join(folder_data, 'period5_k.txt'))


def plot(x1, y1, x2, y2, x3, y3, x4, y4, path):
    fig = plt.figure(figsize=(0.8*fig_width, 1.3*fig_height))
    ax = plt.gca()

    ax.plot(x1, y1, label='axisymmetric expansion')
    ax.scatter(axi_exp_k[:, 0], axi_exp_k[:, 1], marker='o')
    ax.plot(x2, y2, label='axisymmetric contraction')
    ax.scatter(axi_con_k[:, 0], axi_con_k[:, 1], marker='o')
    ax.plot(x3, y3, label='pure shear')
    ax.scatter(shear_k[:, 0], shear_k[:, 1], marker='o')
    ax.plot(x4, y4, label='plain strain')
    ax.scatter(plane_k[:, 0], plane_k[:, 1], marker='o')

    ax.set_xlabel(r'$S\cdot t$')
    ax.set_ylabel(r'$k/k_0$')
    ax.axis(xmin=0, xmax=5, ymin=0, ymax=2.5)
    plt.legend()
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(path, 'compare'))
    plt.close('all')


def plot_b(x1, y1, x2, y2, x4, y4, path):

    fig = plt.figure(figsize=(0.8*fig_width, 1.3*fig_height))
    ax = plt.gca()
    ax.plot(x1, y1, label='axisymmetric expansion')
    ax.scatter(axi_exp_b[:, 0], axi_exp_b[:, 1], marker='^')
    ax.plot(x2, y2, label='axisymmetric contraction')
    ax.scatter(axi_con_b[:, 0], axi_con_b[:, 1], marker='>')
    ax.plot(x4, y4, label='plain strain')
    ax.scatter(plane_b[:, 0], plane_b[:, 1], marker='<')
    ax.set_xlabel(r'$S\cdot t$')
    ax.set_ylabel(r'$k/k_0$')
    # ax.axis(xmin=0, xmax=5, ymin=0, ymax=2.5)
    plt.legend()
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    fig.savefig(os.path.join(path, 'compare_b'))


    plt.close('all')


def plot_periodic(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, path):

    fig = plt.figure(figsize=(1*fig_width, 1.3*fig_height))
    ax = plt.gca()
    ax.plot(x1, y1, label=r'$\omega/S_{max} = 0.125$')
    ax.scatter(period1_k[:, 0], period1_k[:, 1], marker='o')
    ax.plot(x2, y2, label=r'$\omega/S_{max} = 0.25$')
    ax.scatter(period2_k[:, 0], period2_k[:, 1], marker='o')
    ax.plot(x3, y3, label=r'$\omega/S_{max} = 0.5$')
    ax.scatter(period3_k[:, 0], period3_k[:, 1], marker='o')
    ax.plot(x4, y4, label=r'$\omega/S_{max} = 0.75$')
    ax.scatter(period4_k[:, 0], period4_k[:, 1], marker='o')
    ax.semilogy(x5, y5, label=r'$\omega/S_{max} = 1$')
    ax.scatter(period5_k[:, 0], period5_k[:, 1], marker='o')
    ax.set_xlabel(r'$S\cdot t$')
    ax.set_ylabel(r'$k/k_0$')
    ax.axis(xmin=0, ymin=0, xmax=51)
    plt.legend(loc=0, labelspacing=0.2)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(path, 'compare_periodic'))

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
    # cmaplist[0] = 'black'   # force the first color entry to be black
    cmaplist[0] = 'white' # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)

    confidence = np.loadtxt(os.path.join(path['output'], 'confidence'))
    fig = plt.figure(figsize=(1.25*fig_width, 1.1*fig_width))
    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(path['output'], 'marginal_smooth{}'.format(i)))

                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                c_final_smooth = np.loadtxt(os.path.join(os.path.join(path['output'], 'C_final_smooth')))
                ax.axvline(confidence[i, 0], linestyle='--', color='b', label=r'$90\%$ interval')
                ax.axvline(confidence[i, 1], linestyle='--', color='b')
                if len(c_final_smooth.shape) == 1:
                    ax.axvline(c_final_smooth[i], linestyle='--', color='r', label='max of joint pdf')
                elif len(c_final_smooth) < 4:
                    for C in c_final_smooth:
                        ax.axvline(C[i], linestyle='--', color='b', label='joint max')
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)

                if i == 0:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                if i == 1:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                if i == 2:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                if i == 3:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

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

    fig.savefig(os.path.join(path['plots'], 'marginal_smooth'))
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


def main():
    basefolder = './ABC/noise/imp_mcmc/'

    path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots/')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])

    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits'))

    s0 = 3.3
    beta = [0.125, 0.25, 0.5, 0.75, 1]
    ####################################################################################################################
    # plot_marginal_pdf(path, C_limits)
    plot_marginal_smooth_pdf(path, C_limits)

    c = np.loadtxt(os.path.join(path['output'], 'C_final_smooth'))
    print('C_final_smooth: ', c)
    err = np.zeros(13)

    S = utils.axisymmetric_expansion()
    tspan = [0, 1.6 / np.abs(S[0])]
    Tnke1, Ynke1 = m.RK(f=m.rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    err[0] = m.calc_err(np.abs(S[0]) * Tnke1, Ynke1[:, 0], axi_exp_k[:, 0], axi_exp_k[:, 1])
    err[9] = m.calc_err(np.abs(S[0]) * Tnke1, Ynke1[:, 2], axi_exp_b[:, 0], 2 * axi_exp_b[:, 1])
    x1, y1, y1b = np.abs(S[0])*Tnke1, Ynke1[:, 0], Ynke1[:, 2]

    S = utils.axisymmetric_contraction()
    tspan = [0, 1.6 / np.abs(S[0])]
    Tnke2, Ynke2 = m.RK(f=m.rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01 / np.abs(S[0]), args=(c, S))
    err[1] = m.calc_err(np.abs(S[0]) * Tnke2, Ynke2[:, 0], axi_con_k[:, 0], axi_con_k[:, 1])
    err[10] = m.calc_err(np.abs(S[0]) * Tnke2, Ynke2[:, 2], axi_con_b[:, 0], 2 * axi_con_b[:, 1])
    x2, y2, y2b = np.abs(S[0])*Tnke2, Ynke2[:, 0], Ynke2[:, 2]

    S = utils.pure_shear()
    tspan = [0, 5.2/ (2*S[3])]
    Tnke3, Ynke3 = m.RK(f=m.rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    err[2] = m.calc_err(2 * S[3] * Tnke3, Ynke3[:, 0], shear_k[:, 0], shear_k[:, 1])
    x3, y3, y3b = 2*S[3]*Tnke3, Ynke3[:, 0], Ynke3[:, 2]

    S = utils.plane_strain()
    tspan = [0, 1.6/S[0]]
    Tnke4, Ynke4 = m.RK(f=m.rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    err[3] = m.calc_err(np.abs(S[0]) * Tnke4, Ynke4[:, 0], plane_k[:, 0], plane_k[:, 1])
    err[12] = m.calc_err(np.abs(S[0]) * Tnke4, Ynke4[:, 2], plane_b[:, 0], 2 * plane_b[:, 1])
    x4, y4, y4b = 1/2*Tnke4, Ynke4[:, 0], Ynke4[:, 2]

    # # Periodic shear(five different frequencies)
    tspan = np.array([0, 51])/s0
    Tnke5, Ynke5 = m.RK(f=m.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[0]))
    err[4] = m.calc_err(s0 * Tnke5, Ynke5[:, 0], period1_k[:, 0], period1_k[:, 1])
    x5, y5 = s0 * Tnke5, Ynke5[:, 0]
    Tnke6, Ynke6 = m.RK(f=m.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[1]))
    err[5] = m.calc_err(s0 * Tnke6, Ynke6[:, 0], period2_k[:, 0], period2_k[:, 1])
    x6, y6 = s0 * Tnke6, Ynke6[:, 0]
    Tnke7, Ynke7 = m.RK(f=m.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[2]))
    err[6] = m.calc_err(s0 * Tnke7, Ynke7[:, 0], period3_k[:, 0], period3_k[:, 1])
    x7, y7 = s0 * Tnke7, Ynke7[:, 0]
    Tnke8, Ynke8 = m.RK(f=m.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[3]))
    err[7] = m.calc_err(s0 * Tnke8, Ynke8[:, 0], period4_k[:, 0], period4_k[:, 1])
    x8, y8 = s0 * Tnke8, Ynke8[:, 0]
    Tnke9, Ynke9 = m.RK(f=m.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[4]))
    err[8] = m.calc_err(s0 * Tnke9, Ynke9[:, 0], period5_k[:, 0], period5_k[:, 1])
    x9, y9 = s0 * Tnke9, Ynke9[:, 0]

    print('err_k = ', err[:4])
    print('err_k periodic = ', err[4:9])
    print('err_b  = ', err[9:])

    plot(x1, y1, x2, y2, x3, y3, x4, y4, path['plots'])
    plot_b(x1, y1b, x2, y2b, x4, y4b, path['plots'])
    plot_periodic(x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, path['plots'])


if __name__ == '__main__':
    main()