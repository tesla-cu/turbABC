import os

import numpy as np
import plotting.plotting_2Dmarginals as plotting_2Dmarginals
import plot_compare_truth_ransode as plot_compare_truth

def main():
    basefolder = '../../'

    path = {'output': os.path.join(basefolder, 'rans_output'), 'plots': os.path.join(basefolder, 'rans_plots')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    params_names = [r'$C_1$', r'$C_2$', r'$C_{\varepsilon 1}$', r'$C_{\varepsilon 2}$']
    num_bin_kde = 50
    folder = os.path.join(path['output'], 'chains/')
    plot_folder = path['plots']
    print("Plot comparison with true data and kde 2D marginals")
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
    plot_compare_truth.plot_impulsive(c, folder)
    plot_compare_truth.plot_periodic(c, folder)
    plot_compare_truth.plot_decay(c, folder)
    plot_compare_truth.plot_strained(c, folder)
    plot_compare_truth.plot_validation_exp(c, folder)
    plot_compare_truth.plot_validation_nominal(c, folder)
    plotting_2Dmarginals.plot_marginal_pdf(folder, C_limits, C_limits, num_bin_kde, params_names,
                                           plot_folder, smooth="smooth")
    ###################################################################################################################
    print("Plot change of marginal pdfs for different epsilon")


if __name__ == '__main__':
    main()