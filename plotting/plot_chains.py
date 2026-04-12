import glob
import logging
import os
import re
import sys

import numpy as np
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plotting_change_with_eps import plot_MAP_confidence_change, plot_marginal_change


def resolve_num_bin_kde(chains_folder, default=20):
    matches = glob.glob(os.path.join(chains_folder, 'C_final_smooth*'))
    if not matches:
        raise FileNotFoundError(f'{chains_folder} does not contain C_final_smooth* files.')

    bins = []
    for path in matches:
        name = os.path.basename(path)
        match = re.search(r'C_final_smooth(\d+)$', name)
        if match:
            bins.append(int(match.group(1)))

    if not bins:
        raise FileNotFoundError(f'No KDE suffix found in {chains_folder}.')
    return max(bins)


def resolve_params_names(n_params):
    defaults = [
        r'$C_1$',
        r'$C_2$',
        r'$C_{\varepsilon 1}$',
        r'$C_{\varepsilon 2}$',
        r'$C_5$',
        r'$C_6$',
        r'$C_7$',
        r'$C_8$',
        r'$C_9$',
    ]
    return defaults[:n_params]


def find_x_folders(base_folder):
    if not os.path.isdir(base_folder):
        return []
    folders = [
        os.path.join(base_folder, name)
        for name in os.listdir(base_folder)
        if name.startswith('x_') and os.path.isdir(os.path.join(base_folder, name))
    ]
    return sorted(folders)


def main(args=None):
    if args is None:
        args = sys.argv

    if len(args) > 1:
        params_path = args[1]
    else:
        params_path = os.path.join(ROOT, 'les_closure', 'params.yml')

    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    logging.basicConfig(format='%(levelname)s: %(name)s:  %(message)s', level=logging.INFO)

    base_dir = os.path.dirname(os.path.abspath(params_path))
    output_folder = os.path.normpath(os.path.join(base_dir, params['path']['output']))
    plots_folder = os.path.join(base_dir, 'plots')
    calibration_folder = os.path.join(output_folder, 'calibration')
    chains_folder = os.path.join(output_folder, 'chains')

    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)

    c_limits = np.loadtxt(os.path.join(calibration_folder, 'C_limits'))
    n_params = len(c_limits)
    params_names = resolve_params_names(n_params)
    num_bin_kde = resolve_num_bin_kde(chains_folder)

    logging.info('Using chain folder: %s', chains_folder)
    logging.info('Using num_bin_kde=%s', num_bin_kde)

    map_path = os.path.join(chains_folder, f'C_final_smooth{num_bin_kde}')
    c_map = np.loadtxt(map_path)
    logging.info('MAP parameters: %s', c_map)

    regression_folders = [
        os.path.join(output_folder, 'regression_dist'),
        os.path.join(output_folder, 'regression_full'),
    ]

    plotted_any = False
    for regression_folder in regression_folders:
        x_folders = find_x_folders(regression_folder)
        if not x_folders:
            logging.info('Skipping %s: no x_* folders found', regression_folder)
            continue

        logging.info('Plotting epsilon sweep summaries from %s', regression_folder)
        plot_marginal_change(x_folders, params_names, c_limits, num_bin_kde, plots_folder)

        required_confidence = os.path.join(x_folders[0], 'confidence_95')
        required_quantile = os.path.join(x_folders[0], 'quantile_95')
        if os.path.exists(required_confidence) and os.path.exists(required_quantile):
            plot_MAP_confidence_change(x_folders, params_names, num_bin_kde, c_limits, plots_folder)
        else:
            logging.info('Skipping MAP confidence plot for %s: confidence files not found', regression_folder)
        plotted_any = True

    if not plotted_any:
        logging.info('No regression x_* folders were found. Chain summary is available in %s', chains_folder)


if __name__ == '__main__':
    main()
