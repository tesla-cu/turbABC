import glob
import logging
import os
import sys
import yaml
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _save_array_txt(array, out_path):
    """Persist a numpy array to txt with stable formatting."""
    arr = np.asarray(array)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    np.savetxt(out_path, arr)


def convert_npz_to_txt(npz_file, output_folder):
    """Convert each array inside a .npz file into a dedicated .txt file."""
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    with np.load(npz_file) as data:
        for key in data.files:
            txt_name = f"{base_name}_{key}.txt"
            txt_path = os.path.join(output_folder, txt_name)
            _save_array_txt(data[key], txt_path)
            logging.info('saved %s', txt_path)


def main(args):
    if len(args) > 1:
        input_path = args[1]
    else:
        input_path = os.path.join('../les_closure/', 'params.yml')

    with open(input_path, 'r') as f:
        params = yaml.safe_load(f)

    output_path = params['path']['output']
    chains_path = os.path.join(output_path, 'chains')

    logging.basicConfig(
        format='%(levelname)s: %(name)s:  %(message)s',
        handlers=[logging.StreamHandler()],
        level=logging.INFO,
    )

    targets = []
    targets.extend(glob.glob(os.path.join(output_path, '*.npz')))
    targets.extend(glob.glob(os.path.join(chains_path, '*.npz')))
    targets = sorted(set(targets))

    if not targets:
        logging.warning('No .npz files found in %s or %s', output_path, chains_path)
        return

    txt_output = os.path.join(output_path, 'txt_export')
    if not os.path.isdir(txt_output):
        os.makedirs(txt_output)

    logging.info('Converting %d files to txt in %s', len(targets), txt_output)
    for npz_file in targets:
        convert_npz_to_txt(npz_file, txt_output)


if __name__ == '__main__':
    main(sys.argv)
