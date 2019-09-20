import os
import glob

import numpy as np
import plotting
import plot_compare_truth



def main():
    basefolder = '../'

    path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']

    num_bin_kde = 100
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
    data_30 = np.loadtxt(os.path.join(path['output'], '1d_dist_scatter_0.3'))
    print(data_30.shape)
    plotting.plot_regression(data_30, C_limits, params_names[0], path['plots'])
    # plotting.plot_MAP_confidence_change(folders, params_names, num_bin_kde, C_limits, path['plots'])
    # plotting.plot_eps_change(folders, path['plots'])
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
            plot_compare_truth.plot_impulsive(c, reg_plot_folder)
            plot_compare_truth.plot_periodic(c, reg_plot_folder)
            plot_compare_truth.plot_decay(c, reg_plot_folder)
            plot_compare_truth.plot_strained(c, reg_plot_folder)
        print("Plot change of marginal pdfs for different epsilon")
        plotting.plot_1d_pdf_change(folders, params_names, C_limits, num_bin_kde_reg, plot_folder)
    ###################################################################################################################


if __name__ == '__main__':
    main()