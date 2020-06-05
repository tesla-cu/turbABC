import os
import glob
import shutil
# import matplotlib as mpl
# mpl.use('pdf')
import numpy as np
import plotting_2Dmarginals
import plotting
# import plot_compare_truth


def main():
    basefolder = '../overflow_results/'
    nominal_values = None
    ### 4 params
    nominal_values = [0.09, 0.5, 0.075, 0.0828]
    params_names = [r'$\beta^*$', r'$\sigma_{w1}$', r'$\beta_1$', r'$\beta_2$']
    # params_names = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    # params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']
    num_bin_kde = 20
    num_bin_raw = 10

    ### 5 params
    # nominal_values = [0.09, 0.5, 0.075, 0.0828, 0.31]
    # params_names = [r'$\beta^*$', r'$\sigma_{w1}$', r'$\beta_1$', r'$\beta_2$', r'$a_1$']
    nominal_values = [0.09, 0.5, 0.075/0.09, 0.0828/0.09, 0.31]
    params_names = [r'$\beta^*$', r'$\sigma_{w1}$', r'$\beta_1/\beta^*$', r'$\beta_2/\beta^*$', r'$a_1$']
    num_bin_kde = 10
    num_bin_raw = 6


    path = {'output': os.path.join(basefolder, 'output/'), 'plots': os.path.join(basefolder, 'plots')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))

    # stats = ['cp', 'u', 'uv', 'cp_u', 'cp_u_uv', 'x1', 'x2', 'x1_x2', 'all_if_less_05']
    # postprocess_folders = [os.path.join(path['output'], 'postprocess_' + folder) for folder in stats]

    stats = glob.glob1(path['output'], "postprocess_*")     # different statistics
    postprocess_folders = [os.path.join(path['output'], folder) for folder in stats]
    for k, output_folder in enumerate(postprocess_folders):
        print('output: ', output_folder)
        x_list = glob.glob1(output_folder, "x_*")
        folders_x = [os.path.join(output_folder, i) for i in x_list]
        plot_folder = os.path.join(path['plots'], stats[k][12:])
        print("Plot comparison with true data and kde 2D marginals")
        for x, folder in enumerate(folders_x):
            print('folder x: ', folder)
            plot_folder_x = os.path.join(plot_folder, x_list[x])
            if not os.path.isdir(plot_folder_x):
                os.makedirs(plot_folder_x)
            print('plot folder: ', plot_folder_x)
            c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
            print('c = ', c)
            # plot_compare_truth.plot_impulsive(c, plot_folder)
            # plot_compare_truth.plot_periodic(c, plot_folder)
            # plot_compare_truth.plot_decay(c, plot_folder)
            # plot_compare_truth.plot_strained(c, plot_folder)
            plotting.plot_marginal_raw_pdf(folder, C_limits, num_bin_raw, params_names, plot_folder_x)
            plotting_2Dmarginals.plot_marginal_smooth_pdf(folder, C_limits, num_bin_kde, params_names, plot_folder_x)
        ###################################################################################################################
        print("Plot change of marginal pdfs for different epsilon")

        plotting.plot_marginal_change(folders_x, params_names, C_limits, num_bin_kde, plot_folder, nominal_values)
        shutil.copy(os.path.join(plot_folder, 'marginal_change.pdf'),
                    os.path.join(path['plots'], f'marginal_change_{stats[k][12:]}.pdf'))
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