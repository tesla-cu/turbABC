import os
import glob

# import matplotlib as mpl
# mpl.use('pdf')
import numpy as np
import plotting
import plot_compare_truth


def main():
    basefolder = '../runs_abc/'

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
    for folder in folders:
        print(folder)
        plot_folder = os.path.join(folder, 'plots/')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
        print(c)
        plot_compare_truth.plot_impulsive(c, plot_folder)
        plot_compare_truth.plot_periodic(c, plot_folder)
        plot_compare_truth.plot_decay(c, plot_folder)
        plot_compare_truth.plot_strained(c, plot_folder)
        plotting.plot_marginal_smooth_pdf(folder, C_limits, num_bin_kde, params_names, plot_folder)
    ###################################################################################################################
    print("Plot change of marginal pdfs for different epsilon")
    plotting.plot_marginal_change(folders, params_names, C_limits, num_bin_kde, path['plots'])
    plotting.plot_MAP_confidence_change(folders, params_names, num_bin_kde, C_limits, path['plots'])
    plotting.plot_eps_change(folders, path['plots'])
    ###################################################################################################################


    # num_bin_kde_reg = 20
    # path['regression'] = os.path.join(path['output'], 'regression/')
    # folders_reg = glob.glob1(path['regression'], "x_*")
    # print(folders_reg)
    # folders = [os.path.join(path['regression'], i) for i in folders_reg]
    # for folder in folders:
    #     print(folder)
    #     reg_plot_folder = os.path.join(folder, 'plots/')
    #     if not os.path.isdir(reg_plot_folder):
    #         os.makedirs(reg_plot_folder)
    #     limits = np.loadtxt(os.path.join(folder, 'reg_limits'))
    #     print('Plotting regression marginal pdf')
    #     plotting.plot_marginal_smooth_pdf(folder, limits, num_bin_kde_reg, params_names, reg_plot_folder)
    #     print("Plot comparison with true data and kde 2D marginals")
    #     c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde_reg)))
    #     print("C_MAP = ", c)
    #     # plot_compare_truth.plot_impulsive(c, reg_plot_folder)
    #     # plot_compare_truth.plot_periodic(c, reg_plot_folder)
    #     # plot_compare_truth.plot_decay(c, reg_plot_folder)
    #     # plot_compare_truth.plot_strained(c, reg_plot_folder)


if __name__ == '__main__':
    main()