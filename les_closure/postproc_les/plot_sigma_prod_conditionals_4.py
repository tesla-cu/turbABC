import glob
import logging
import os
import shutil
import sys

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import numpy as np
import yaml


LES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

single_column = 252
oneandhalf_column = 397
double_column = 522
text_height = 682
PARAM_NAMES = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
N_PARAMS = 4


def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = 0.45 * text_height * inches_per_pt
    return fig_width, fig_height


mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8 * plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8 * plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 1


def resolve_default_paths(params_path):
    with open(params_path, 'r') as stream:
        params = yaml.safe_load(stream)

    base_dir = os.path.dirname(os.path.abspath(params_path))
    output_root = os.path.normpath(os.path.join(base_dir, params['path']['output']))
    return {
        'params': params,
        'base_dir': base_dir,
        'output_root': output_root,
        'chains': os.path.join(output_root, 'chains'),
        'calibration': os.path.join(output_root, 'calibration'),
        'plots': os.path.join(base_dir, 'plots'),
    }


def normalize_dataset_folder(path):
    if os.path.isfile(os.path.join(path, 'marginal_smooth0')):
        return path

    chains_path = os.path.join(path, 'chains')
    if os.path.isfile(os.path.join(chains_path, 'marginal_smooth0')):
        return chains_path

    raise FileNotFoundError(
        f'{path} does not contain postprocessed marginal files. '
        'Pass a chains folder or an output folder containing a chains subfolder.'
    )


def ensure_c_final_smooth(folder):
    bare_path = os.path.join(folder, 'C_final_smooth')
    if os.path.isfile(bare_path):
        return bare_path

    candidates = [path for path in glob.glob(os.path.join(folder, 'C_final_smooth*')) if os.path.isfile(path)]
    if not candidates:
        raise FileNotFoundError(f'No C_final_smooth files found in {folder}. Run postprocess/postproc_chains.py first.')

    latest_path = max(candidates, key=os.path.getmtime)
    shutil.copyfile(latest_path, bare_path)
    logging.info('Created compatibility file %s from %s', bare_path, latest_path)
    return bare_path


def load_limits(folder, defaults, calibration_folder=None):
    for file_name in ['reg_limits', 'C_limits']:
        candidate = os.path.join(folder, file_name)
        if os.path.isfile(candidate):
            return np.atleast_2d(np.loadtxt(candidate))[:N_PARAMS]

    if calibration_folder is not None:
        candidate = os.path.join(calibration_folder, 'C_limits')
        if os.path.isfile(candidate):
            return np.atleast_2d(np.loadtxt(candidate))[:N_PARAMS]

    return np.asarray(defaults['C_limits'], dtype=float)[:N_PARAMS]


def load_dataset(folder, defaults, calibration_folder=None):
    normalized_folder = normalize_dataset_folder(folder)
    limits = load_limits(normalized_folder, defaults, calibration_folder)
    c_final = np.atleast_1d(np.loadtxt(ensure_c_final_smooth(normalized_folder)))

    marginals = {}
    for index in range(N_PARAMS):
        marginal = np.loadtxt(os.path.join(normalized_folder, f'marginal_smooth{index}'))
        spacing = marginal[0, 1] - marginal[0, 0]
        norm = np.sum(marginal[1]) * spacing
        marginal[1] /= norm
        marginals[str(index)] = marginal

    conditionals = {}
    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            if i == j:
                continue
            candidate = os.path.join(normalized_folder, f'conditional_smooth{i}{j}')
            if not os.path.isfile(candidate):
                continue
            conditional = np.loadtxt(candidate)
            n_bins = conditional.shape[0] - 1
            norm = np.sum(conditional) * (limits[j, 1] - limits[j, 0]) * (limits[i, 1] - limits[i, 0]) / n_bins ** 2
            conditionals[f'{i}{j}'] = conditional / norm

    return {
        'folder': normalized_folder,
        'limits': limits,
        'c_final': c_final,
        'marginals': marginals,
        'conditionals': conditionals,
    }


def combine_limits(*datasets):
    combined = np.zeros((N_PARAMS, 2))
    for index in range(N_PARAMS):
        combined[index, 0] = min(dataset['limits'][index, 0] for dataset in datasets)
        combined[index, 1] = max(dataset['limits'][index, 1] for dataset in datasets)
    combined[0, 1] = min(0.0, combined[0, 1])
    return combined


def compute_max_value(*datasets):
    max_value = 0.0
    for dataset in datasets:
        for values in dataset['conditionals'].values():
            max_value = max(max_value, np.max(values))
    return max_value


def choose_dataset_folders(args, defaults):
    fallback = defaults['chains']
    sigma_folder = args[1] if len(args) > 1 else fallback
    prod_folder = args[2] if len(args) > 2 else fallback
    both_folder = args[3] if len(args) > 3 else fallback
    output_path = args[4] if len(args) > 4 else os.path.join(defaults['plots'], 'conditional_combined_4.pdf')
    return sigma_folder, prod_folder, both_folder, output_path


def validate_datasets(sigma, prod):
    required_keys = [f'{i}{j}' for i in range(N_PARAMS) for j in range(N_PARAMS) if i != j]
    for key in required_keys:
        if key not in sigma['conditionals']:
            raise FileNotFoundError(f'Missing {key} conditional in {sigma["folder"]}.')
        if key not in prod['conditionals']:
            raise FileNotFoundError(f'Missing {key} conditional in {prod["folder"]}.')


def main(args=None):
    if args is None:
        args = sys.argv

    params_path = os.path.join(LES_ROOT, 'params.yml')
    defaults = resolve_default_paths(params_path)
    sigma_folder, prod_folder, both_folder, output_path = choose_dataset_folders(args, defaults)

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sigma = load_dataset(sigma_folder, defaults['params'], defaults['calibration'])
    prod = load_dataset(prod_folder, defaults['params'], defaults['calibration'])
    both = load_dataset(both_folder, defaults['params'], defaults['calibration'])
    validate_datasets(sigma, prod)

    combined_limits = combine_limits(sigma, prod, both)
    max_value = compute_max_value(sigma, prod)

    cmap = cm.inferno
    cmaplist = [cmap(i) for i in reversed(range(cmap.N))]
    gamma = 1 / np.e
    cmaplist[0] = 'white'
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist)

    fig_width, fig_height = fig_size(width_column=double_column)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ticks = [0.03, 0.25, 0.2, 0.2]

    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            if i == j:
                ax = plt.subplot2grid((N_PARAMS, N_PARAMS), (i, i))
                ax.plot(sigma['marginals'][str(i)][0], sigma['marginals'][str(i)][1], 'b', label='sigma')
                ax.axvline(sigma['c_final'][i], linestyle='--', color='b')
                ax.plot(prod['marginals'][str(i)][0], prod['marginals'][str(i)][1], 'r', label='production')
                ax.axvline(prod['c_final'][i], linestyle='--', color='r')
                ax.plot(both['marginals'][str(i)][0], both['marginals'][str(i)][1], 'g', label='both')
                ax.axvline(both['c_final'][i], linestyle='--', color='g')
                ax.axis(xmin=combined_limits[i, 0], xmax=combined_limits[i, 1], ymin=0)
                ax.tick_params(axis='both', which='major', pad=2)
                ax.tick_params(axis='y', which='major', pad=0.2)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                ax.grid(True, linestyle='--', alpha=0.5)
                if i != (N_PARAMS - 1):
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_xlabel(PARAM_NAMES[i], labelpad=2)
                if i == 0:
                    ax.set_ylabel('pdf')

            elif i < j:
                ax = plt.subplot2grid((N_PARAMS, N_PARAMS), (i, j))
                ax.axis(xmin=combined_limits[j, 0], xmax=combined_limits[j, 1], ymin=combined_limits[i, 0], ymax=combined_limits[i, 1])
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                ax.tick_params(axis='both', which='major', pad=1.5)
                ax.yaxis.tick_right()
                ax.grid(True, linestyle='--', alpha=0.5)
                if j != N_PARAMS - 1:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_ylabel(PARAM_NAMES[i])
                    ax.yaxis.set_label_position('right')

                ext = (sigma['limits'][j, 0], sigma['limits'][j, 1], sigma['limits'][i, 0], sigma['limits'][i, 1])
                ax.imshow(
                    sigma['conditionals'][f'{j}{i}'],
                    origin='lower',
                    cmap=cmap,
                    aspect='auto',
                    extent=ext,
                    norm=colors.PowerNorm(gamma=gamma, vmin=0, vmax=max_value),
                )
                ax.axvline(sigma['c_final'][j], linestyle='--', color='b')
                ax.axhline(sigma['c_final'][i], linestyle='--', color='b')

            else:
                ax = plt.subplot2grid((N_PARAMS, N_PARAMS), (i, j))
                ax.axis(xmin=combined_limits[j, 0], xmax=combined_limits[j, 1], ymin=combined_limits[i, 0], ymax=combined_limits[i, 1])
                ax.tick_params(axis='both', which='major', pad=2)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
                ax.grid(True, linestyle='--', alpha=0.5)
                if j == 0:
                    ax.set_ylabel(PARAM_NAMES[i])
                else:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                if i != (N_PARAMS - 1):
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                else:
                    ax.set_xlabel(PARAM_NAMES[j], labelpad=2)

                ext = (prod['limits'][j, 0], prod['limits'][j, 1], prod['limits'][i, 0], prod['limits'][i, 1])
                im_cond = ax.imshow(
                    prod['conditionals'][f'{i}{j}'].T,
                    origin='lower',
                    cmap=cmap,
                    aspect='auto',
                    extent=ext,
                    norm=colors.PowerNorm(gamma=gamma, vmin=0, vmax=max_value),
                )
                ax.axvline(prod['c_final'][j], linestyle='--', color='r')
                ax.axhline(prod['c_final'][i], linestyle='--', color='r')

    ticks_cbar = np.arange(0, 160, 35)[1:]
    cax = plt.axes([0.81, 0.08, 0.01, 0.21])
    plt.colorbar(im_cond, cax=cax, ticks=ticks_cbar)
    fig.subplots_adjust(left=0.2, right=0.8, wspace=0.11, hspace=0.11, bottom=0.08, top=0.98)
    fig.savefig(output_path)
    plt.close('all')


if __name__ == '__main__':
    main(sys.argv)
