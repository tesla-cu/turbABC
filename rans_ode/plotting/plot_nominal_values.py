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
    nominal_values = np.array([1.5, 0.8, 1.44, 1.92])
    path = {'plots': os.path.join(basefolder, 'rans_plots', 'nominal')}
    if not os.path.isdir(path['plots']):
        os.makedirs(path['plots'])
    plot_compare_truth.plot_impulsive(nominal_values, path['plots'])
    plot_compare_truth.plot_periodic(nominal_values, path['plots'])
    plot_compare_truth.plot_decay(nominal_values, path['plots'])
    plot_compare_truth.plot_strained(nominal_values, path['plots'])
    plot_compare_truth.plot_validation_exp(nominal_values, path['plots'])

if __name__ == '__main__':
    main()