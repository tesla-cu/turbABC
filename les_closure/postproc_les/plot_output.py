import glob
import logging
import os
import shutil
import sys

import numpy as np
import yaml


LES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if LES_ROOT not in sys.path:
    sys.path.insert(0, LES_ROOT)

import postproc_les.plotting as plotting


def resolve_paths(params_path, params):
    base_dir = os.path.dirname(os.path.abspath(params_path))
    output_root = os.path.normpath(os.path.join(base_dir, params['path']['output']))
    return {
        'output_root': output_root,
        'chains': os.path.join(output_root, 'chains'),
        'calibration': os.path.join(output_root, 'calibration'),
        'plots': os.path.join(base_dir, 'plots'),
    }


def load_c_limits(paths, params):
    calibration_limits = os.path.join(paths['calibration'], 'C_limits')
    if os.path.isfile(calibration_limits):
        return np.atleast_2d(np.loadtxt(calibration_limits))
    return np.asarray(params['C_limits'], dtype=float)[:params['model']['N_params']]


def ensure_chain_compatibility_files(chains_folder):
    pattern = os.path.join(chains_folder, 'C_final_smooth*')
    candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not candidates:
        raise FileNotFoundError(f'No C_final_smooth files found in {chains_folder}. Run postprocess/postproc_chains.py first.')

    bare_path = os.path.join(chains_folder, 'C_final_smooth')
    if os.path.isfile(bare_path):
        return bare_path

    latest_path = max(candidates, key=os.path.getmtime)
    shutil.copyfile(latest_path, bare_path)
    logging.info('Created compatibility file %s from %s', bare_path, latest_path)
    return bare_path


def plot_postprocessed_marginals(paths, c_limits, n_params):
    if not os.path.isdir(paths['chains']):
        raise FileNotFoundError(f"{paths['chains']} not found. Run postprocess/postproc_chains.py first.")

    ensure_chain_compatibility_files(paths['chains'])
    os.makedirs(paths['plots'], exist_ok=True)

    if n_params == 3:
        plotting.plot_marginal_smooth_pdf_3(n_params, paths['chains'], paths['plots'], c_limits)
        return
    if n_params == 4:
        plotting.plot_marginal_smooth_pdf_4(n_params, paths['chains'], paths['plots'], c_limits)
        return

    raise NotImplementedError(f'plot_output.py currently supports 3 or 4 parameters, got {n_params}.')


def main(args=None):

    params_path = os.path.join(LES_ROOT, 'params.yml')

    with open(params_path, 'r') as stream:
        params = yaml.safe_load(stream)

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    paths = resolve_paths(params_path, params)
    c_limits = load_c_limits(paths, params)
    n_params = int(params['model']['N_params'])

    if len(c_limits) < n_params:
        raise ValueError(f'Expected at least {n_params} parameter bounds, found {len(c_limits)}.')

    plot_postprocessed_marginals(paths, c_limits[:n_params], n_params)


if __name__ == '__main__':
    main(sys.argv)