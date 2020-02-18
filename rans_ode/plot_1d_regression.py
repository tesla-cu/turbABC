import os
import glob

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import plotting
import plot_compare_truth_ransode


def plot_MAP_comparison(data_folders, data_folders_reg, plot_folder, C_limits):
    MAP, MAP_reg = [], []
    x = np.empty(len(data_folders))
    for i, folder in enumerate(data_folders):
        x[i] = float(os.path.basename(os.path.normpath(folder))[2:])
        print('x = ', x[i])
        MAP.append(np.loadtxt(os.path.join(folder, 'C_final_smooth')))
    for folder in data_folders_reg:
        MAP_reg.append(np.loadtxt(os.path.join(folder, 'C_final_smooth')))
    ind = np.argsort(x)
    x = x[ind]
    MAP = np.array(MAP)[ind]
    MAP_reg = np.array(MAP_reg)[ind]
    print(MAP)
    print(MAP_reg)
    colors = ['b', 'g', 'y']
    colors2 = ['orange', 'k', 'magenta']
    fig = plt.figure()
    ax = plt.gca()
    for j in range(len(x)):
        MAP[j] = MAP[j].reshape((-1, 1))
        MAP_reg[j] = MAP_reg[j].reshape((-1, 1))
        if len(MAP[j])>1:
            for map in MAP[j]:
                ax.scatter(map, x[j], s=10, color='r', zorder=2)
        else:
            ax.scatter(MAP[j], x[j], s=10, color='r', zorder=2)

        ax.scatter(MAP_reg[j], x[j], s=10, color='k', zorder=2)
    ax.set_xlabel('C')
    ax.set_xlim([1.2, 1.7])
    ax.set_yscale('log')
    fig.savefig(os.path.join(plot_folder, 'MAP_change'))
    plt.close('all')


def main():
    basefolder = '../'

    path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']

    num_bin_kde = 20
    folders_abc = glob.glob1(path['output'], "x_*")
    print(folders_abc)
    folders = [os.path.join(path['output'], i) for i in folders_abc]
    print("Plot comparison with true data and kde 2D marginals")
    eps_list = []
    for f, folder in enumerate(folders):
        print(folder)
        plot_folder = os.path.join(path['plots'], folders_abc[f])
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        print(plot_folder)
        c = np.array([np.loadtxt(os.path.join(folder, 'C_final_smooth')), 0.8, 1.44, 1.92])
        eps_list.append(np.loadtxt(os.path.join(folder, 'eps')))
        print('c = ', c)
        # plot_compare_truth.plot_impulsive(c, plot_folder)
        # plot_compare_truth.plot_periodic(c, plot_folder)
        # plot_compare_truth.plot_decay(c, plot_folder)
        # plot_compare_truth.plot_strained(c, plot_folder)
    ###################################################################################################################
    print("Plot change of marginal pdfs for different epsilon")
    plotting.plot_1d_pdf_change(folders, params_names, C_limits, num_bin_kde, path['plots'])
    data = np.loadtxt(os.path.join(path['output'], '1d_dist_scatter'))
    print(folders_abc)
    print(eps_list)
    plotting.plot_1d_dist_scatter(data, C_limits, params_names[0], folders_abc, eps_list, path['plots'])
    plotting.plot_sampling_hist(data[:, 0], C_limits, params_names[0], path['plots'])
    data_10 = np.load(os.path.join(path['output'], '1d_dist_scatter_0.1.npz'))
    solution = np.empty((data_10['sumstat'].shape[1], 2))
    for i in range(data_10['sumstat'].shape[1]):
        solution[i] = np.loadtxt(os.path.join(path['output'], 'regression_full/x_10.0/solution{}'.format(i)))
    plotting.plot_regression(data_10['C'], data_10['sumstat'], data_10['dist'], solution, params_names[0], path['plots'])

    # plotting.plot_eps_change(folders, path['plots'])
    folders1 = folders
    ###################################################################################################################
    reg_type = ['regression_dist', 'regression_full']

    num_bin_kde_reg = 20
    for type in reg_type:
        path['regression'] = os.path.join(path['output'], type)
        folders_reg = glob.glob1(path['regression'], "x_*")
        print(folders_reg)
        folders = [os.path.join(path['regression'], i) for i in folders_reg]
        plot_folder = os.path.join(path['plots'], type)
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        for f, folder in enumerate(folders):
            print(folder)
            reg_plot_folder = os.path.join(plot_folder, folders_reg[f])
            if not os.path.isdir(reg_plot_folder):
                os.makedirs(reg_plot_folder)
            limits = np.loadtxt(os.path.join(folder, 'reg_limits'))
            print("Plot comparison with true data and kde 2D marginals")
            c = np.array([np.loadtxt(os.path.join(folder, 'C_final_smooth')), 0.8, 1.44, 1.92])
            print("C_MAP = ", c)
            plot_compare_truth_ransode.plot_impulsive(c, reg_plot_folder)
            plot_compare_truth_ransode.plot_periodic(c, reg_plot_folder)
            plot_compare_truth_ransode.plot_decay(c, reg_plot_folder)
            plot_compare_truth_ransode.plot_strained(c, reg_plot_folder)
        print("Plot change of marginal pdfs for different epsilon")
        plotting.plot_1d_pdf_change(folders, params_names, C_limits, num_bin_kde_reg, plot_folder)
        folders2 = folders
    ###################################################################################################################
    plot_MAP_comparison(folders1, folders2, path['plots'], C_limits)

if __name__ == '__main__':
    main()