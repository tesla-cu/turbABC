import os
import glob

# import matplotlib as mpl
# mpl.use('pdf')
import numpy as np
import plotting_2Dmarginals
import plotting
# import plot_compare_truth


def main():
    basefolder = '../'

    path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    # params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']
    params_names = [r'$\beta^*$', r'$\sigma_{w1}$', r'$\beta_1$', r'$\beta_2$']

    num_bin_kde = 20
    folders_abc = glob.glob1(path['output'], "x_*")
    print(folders_abc)
    folders = [os.path.join(path['output'], i) for i in folders_abc]
    print("Plot comparison with true data and kde 2D marginals")
    for f, folder in enumerate(folders):
        print(folder)
        plot_folder = os.path.join(path['plots'], folders_abc[f])
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        print(plot_folder)
        c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
        print(c)
        # plot_compare_truth.plot_impulsive(c, plot_folder)
        # plot_compare_truth.plot_periodic(c, plot_folder)
        # plot_compare_truth.plot_decay(c, plot_folder)
        # plot_compare_truth.plot_strained(c, plot_folder)
        # plotting.plot_marginal_raw_pdf(folder, C_limits, num_bin_kde, params_names, plot_folder)
        plotting_2Dmarginals.plot_marginal_smooth_pdf(folder, C_limits, num_bin_kde, params_names, plot_folder)
    ###################################################################################################################
    # print("Plot change of marginal pdfs for different epsilon")
    nominal_values = [0.09, 0.5, 0.075, 0.0828]
    plotting.plot_marginal_change(folders, params_names, C_limits, num_bin_kde, path['plots'], nominal_values)
    # plotting.plot_MAP_confidence_change(folders, params_names, num_bin_kde, C_limits, path['plots'])
    # plotting.plot_eps_change(folders, path['plots'])
    # folder_reg = os.path.join(path['output'], 'regression_full/x_30.0')
    # plotting.plot_marginal_change_with_regression(folders, folder_reg, params_names, C_limits, num_bin_kde, path['plots'])
    ###################################################################################################################


    # num_bin_kde_reg = 100
    # regression_type = ['regression_dist', 'regression_full']
    # for type in regression_type:
    #     path[type] = os.path.join(path['output'], type)
    #     plot_folder_base = os.path.join(path['plots'], type)
    #     if not os.path.isdir(plot_folder_base):
    #         os.makedirs(plot_folder_base)
    #     folders_reg = glob.glob1(path[type], "x_*")
    #     print(folders_reg)
    #     folders = [os.path.join(path[type], i) for i in folders_reg]
    #     for f, folder in enumerate(folders):
    #         print(folder)
    #         plot_folder = os.path.join(plot_folder_base, folders_reg[f])
    #         if not os.path.isdir(plot_folder):
    #             os.makedirs(plot_folder)
    #         limits = np.loadtxt(os.path.join(folder, 'reg_limits'))
    #         print('Plotting regression marginal pdf')
    #         plotting.plot_marginal_smooth_pdf(folder, limits, num_bin_kde_reg, params_names, plot_folder)
    #         print("Plot comparison with true data and kde 2D marginals")
    #         c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde_reg)))
    #         print("C_MAP = ", c)
    #         # plot_compare_truth.plot_impulsive(c, plot_folder)
    #         # plot_compare_truth.plot_periodic(c, plot_folder)
    #         # plot_compare_truth.plot_decay(c, plot_folder)
    #         # plot_compare_truth.plot_strained(c, plot_folder)
    #     ###################################################################################################################
    #     print("Plot change of marginal pdfs for different epsilon")
    #     plotting.plot_marginal_change(folders, params_names, C_limits, num_bin_kde, plot_folder_base)
    #     # plotting.plot_MAP_confidence_change(folders, params_names, num_bin_kde, C_limits, plot_folder_base)
    #     ###################################################################################################################

if __name__ == '__main__':
    main()