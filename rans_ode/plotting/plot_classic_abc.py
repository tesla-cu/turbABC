import os
import glob
import shutil
# import matplotlib as mpl
# mpl.use('pdf')
import numpy as np
import plotting.plotting_2Dmarginals as plotting_2Dmarginals
import plotting.plotting_change_with_eps as plotting_change_with_eps
import plot_compare_truth_ransode as plot_compare_truth


def main():
    basefolder = '../../'

    ### rans_ode
    nominal_values = [1.5, 0.8, 1.44, 1.92]
    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']
    num_bin_kde = 100
    num_bin_raw = [60]*4

    path = {'output': os.path.join(basefolder, 'rans_output/'),
            'plots': os.path.join(basefolder, 'rans_plots_exp/')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    plot_compare_truth.plot_experiment(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    limits = C_limits
    stats = glob.glob1(path['output'], "postprocess*")     # different statistics
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
            # plot_compare_truth.plot_impulsive(c, plot_folder_x)
            plot_compare_truth.plot_periodic(c, plot_folder_x)
            # plot_compare_truth.plot_decay(c, plot_folder_x)
            # plot_compare_truth.plot_strained(c, plot_folder_x)
            plot_compare_truth.plot_validation_exp(c, plot_folder_x)
            plot_compare_truth.plot_validation_nominal(c, plot_folder_x)
            plotting_2Dmarginals.plot_marginal_pdf(folder, C_limits, C_limits, num_bin_raw, params_names, plot_folder_x)
            plotting_2Dmarginals.plot_marginal_pdf(folder, C_limits, C_limits, num_bin_kde, params_names,
                                                   plot_folder_x, smooth="smooth")
        ###################################################################################################################
        print("Plot change of marginal pdfs for different epsilon")
        plotting_change_with_eps.plot_marginal_change(folders_x, params_names, C_limits, 0, plot_folder, nominal_values)
        plotting_change_with_eps.plot_marginal_change(folders_x, params_names, C_limits, num_bin_kde,
                                                      plot_folder, nominal_values, smooth='smooth')
        shutil.copy(os.path.join(plot_folder, 'marginal_change_smooth.pdf'),
                    os.path.join(path['plots'], f'marginal_change_{stats[k][12:]}.pdf'))
        # plotting.plot_MAP_confidence_change(folders, params_names, num_bin_kde, C_limits, path['plots'])
        # plotting.plot_eps_change(folders, path['plots'])
        # folder_reg = os.path.join(path['output'], 'regression_full/x_30.0')
        # plotting.plot_marginal_change_with_regression(folders, folder_reg, params_names, C_limits, num_bin_kde, path['plots'])
        ###################################################################################################################

        #
        # num_bin_kde_reg = 20
        # regression_type = ['regression_dist']
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
        #         # plotting.plot_marginal_smooth_pdf(folder, limits, num_bin_kde_reg, params_names, plot_folder)
        #         plotting_2Dmarginals.plot_marginal_pdf(folder, limits, limits, num_bin_kde_reg, params_names,
        #                                                plot_folder, smooth="smooth")
        #         print("Plot comparison with true data and kde 2D marginals")
        #         c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde_reg)))
        #         print("C_MAP = ", c)
        #         # plot_compare_truth.plot_impulsive(c, plot_folder)
        #         # plot_compare_truth.plot_periodic(c, plot_folder)
        #         # plot_compare_truth.plot_decay(c, plot_folder)
        #         # plot_compare_truth.plot_strained(c, plot_folder)
        #     ###################################################################################################################
        #     print("Plot change of marginal pdfs for different epsilon")
        #     # plotting.plot_marginal_change(folders, params_names, C_limits, num_bin_kde, plot_folder_base)
        #     plotting_change_with_eps.plot_marginal_change(folders, params_names, limits, num_bin_kde_reg,
        #                                                   plot_folder, nominal_values, smooth='smooth')
        #     # plotting.plot_MAP_confidence_change(folders, params_names, num_bin_kde, C_limits, plot_folder_base)
        #     ###################################################################################################################

if __name__ == '__main__':
    main()