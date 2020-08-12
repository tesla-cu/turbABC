import os

import numpy as np
import plotting.plotting_2Dmarginals as plotting_2Dmarginals


def main():
    basefolder = '../../overflow_results/'

    path = {'output': os.path.join(basefolder, 'chains_limits_final'), 'plots': os.path.join(basefolder, 'plots_limits')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    nominal_values = [0.09, 0.075/0.09, 0.0828/0.09, 0.31]
    params_names = [r'$\beta^*$', r'$\beta_1/\beta^*$', r'$\beta_2/\beta^*$', r'$a_1$']
    num_bin_kde = 50
    folder = os.path.join(path['output'], 'postprocess/')
    plot_folder = path['plots']
    print("Plot comparison with true data and kde 2D marginals")
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    c = np.loadtxt(os.path.join(folder, 'C_final_smooth{}'.format(num_bin_kde)))
    plotting_2Dmarginals.plot_marginal_pdf(folder, C_limits, C_limits, num_bin_kde, params_names,
                                           plot_folder, smooth="smooth")
    ###################################################################################################################
    print("Plot change of marginal pdfs for different epsilon")


if __name__ == '__main__':
    main()