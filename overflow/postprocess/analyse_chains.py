import os
import string
import numpy as np
import pyabc.kde as kde
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors


plt.style.use('dark_background')
# dessertation size
single_column = 235
oneandhalf_column = 352
double_column = 470
def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height

N_PARAMS = 4
N_bins = 20

def autocorr(x):
    result = np.correlate(x, x, mode='same')
    denom = np.sum(x**2)
    return result[len(x)//2:]/denom


def chain_autocorrelation(chain):

    length = len(chain)//2 + len(chain)%2
    print('length', length)
    autocorrelation = np.empty((N_PARAMS, length))
    for i in range(N_PARAMS):
        autocorrelation[i] = autocorr(chain[:, i])
    return autocorrelation


def calc_kde_marginal(c_array, C_limits):
    Z = kde.gaussian_kde_scipy(c_array, C_limits[:, 0], C_limits[:, 1], N_bins)
    map_value_all = kde.find_MAP_kde(Z, C_limits[:, 0], C_limits[:, 1])[0]
    marginal = np.empty((4, N_bins+1))
    for i in range(N_PARAMS):
        ind = tuple(np.where(np.arange(N_PARAMS) != i)[0])
        marginal[i] = np.sum(Z, axis=ind)
        x = np.linspace(C_limits[i, 0], C_limits[i, 1], Z.shape[0])
        marginal[i] = marginal[i] / np.sum(marginal[i]) / np.diff(x)[0]
    return marginal


def calc_marginal_kde(c_array, C_limits):
    # pdf_all = kde.gaussian_kde_scipy(c_array, C_limits[:, 0], C_limits[:, 1], N_bins)
    # map_value_all = kde.find_MAP_kde(pdf_all, C_limits[:, 0], C_limits[:, 1])[0]
    marginal = np.empty((4, N_bins+1))
    for i in range(N_PARAMS):
        marginal[i] = kde.gaussian_kde_scipy(c_array[:, i], C_limits[i, 0], C_limits[i, 1], N_bins)
        x = np.linspace(C_limits[i, 0], C_limits[i, 1], N_bins)
        marginal[i] = marginal[i] / np.sum(marginal[i]) / np.diff(x)[0]
    return marginal


def MAP_batches(c_array, n_batches, C_limits):
    batch_size = len(c_array)//n_batches
    map_values = np.empty((n_batches, N_PARAMS))
    for i in range(n_batches):
        batch = c_array[i*batch_size:(i+1)*batch_size]
        pdf = kde.gaussian_kde_scipy(batch, C_limits[:, 0], C_limits[:, 1], N_bins)
        map_value = kde.find_MAP_kde(pdf, C_limits[:, 0], C_limits[:, 1])[0]
        map_values[i] = map_value
    return map_values, np.mean(map_values, axis=0), np.std(map_values, axis=0)


def main():
    basefolder = '../../overflow_results'

    chains_folder = os.path.join(basefolder, 'chains_limits_final', )
    plot_folder = os.path.join(chains_folder, 'plots')
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    print(chains_folder)
    N_chains = 200
    chain_files = [os.path.join(chains_folder, f'chain{i}.npz') for i in range(N_chains)]
    C_limits = np.load(chain_files[0])['C_limits']
    chains_length = []
    acorr_all = []
    kde_marginals = np.empty((N_chains, N_PARAMS, N_bins+1))
    marginals_kde = np.empty((N_chains, N_PARAMS, N_bins+1))
    for i, file in enumerate(chain_files):
        c_array = np.load(file)['c_array']
        chains_length.append(len(c_array))
        print(i, chains_length[-1])
        kde_marginals[i] = calc_kde_marginal(c_array, C_limits)
        marginals_kde[i] = calc_marginal_kde(c_array, C_limits)
        acorr_all.append(chain_autocorrelation(c_array))
        # map_values, map_mean, map_std = MAP_batches(c_array, 10, C_limits)
        # print(i, map_mean, map_std)
    grid_x = [np.linspace(C_limits[i, 0], C_limits[i, 1], N_bins + 1) for i in range(N_PARAMS)]
    print(np.max(chains_length), np.min(chains_length), np.mean(chains_length), np.std(chains_length))
    np.savez(chains_folder, 'chains_analysis', kde_matginals=kde_marginals, marginals_kde=marginals_kde, grid_x=grid_x,
             chains_length=chains_length)

    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=1, ncols=N_PARAMS, figsize=(fig_width, 0.8 * fig_height))
    for j, chain_marginals in enumerate(marginals_kde):
        for i, marg in enumerate(chain_marginals):
            axarr[i].plot(grid_x[i], marg)
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.1, hspace=0.1, bottom=0.18, top=0.82)
    fig.savefig(os.path.join(plot_folder, f'marginal_chains'))
    plt.close('all')

    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=N_PARAMS, ncols=1, figsize=(fig_width, 4*fig_height))
    for j, acorr_chain in enumerate(acorr_all):
        for i, acorr in enumerate(acorr_chain):
            axarr[i].plot(acorr)
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.1, hspace=0.1, bottom=0.1, top=0.92)
    fig.savefig(os.path.join(plot_folder, f'acorr_chains'))
    plt.close('all')

    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=N_PARAMS, ncols=1, figsize=(fig_width, 3*fig_height))
    for i, c in enumerate(c_array.T):
        axarr[i].plot(c)
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.1, hspace=0.1, bottom=0.1, top=0.92)
    fig.savefig(os.path.join(plot_folder, f'chains'))
    plt.close('all')

if __name__ == '__main__':
    main()