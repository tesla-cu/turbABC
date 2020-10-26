import logging
import numpy as np
import os
import sys
import glob
from postprocess.postprocess_classic_abc import output_by_percent



def load_c(files, N_params):
    accepted = np.empty((0, N_params))
    for file in files:
        logging.debug('loading C from {}'.format(file))
        accepted = np.vstack((accepted, np.load(file)['C'][:, :N_params]))
    return accepted


def load_sumstat(files):
    sum_stat = np.empty((0, len(np.load(files[0])['sumstat'][0])))
    for file in files:
        logging.debug('loading sum_stat from {}'.format(file))
        sum_stat = np.vstack((sum_stat, np.load(file)['sumstat']))
    return sum_stat


def load_dist(files):
    dist = np.empty((0, 1))
    for file in files:
        logging.debug('loading dist from {}'.format(file))
        dist = np.vstack((dist, np.load(file)['dist'].reshape((-1, 1))))
    return dist


def new_limits(new_samples, N_params):
    limits = np.empty((N_params, 2))
    for i in range(N_params):
        limits[i, 0] = np.min(new_samples[:, i])
        limits[i, 1] = np.max(new_samples[:, i])
        if limits[i, 1] - limits[i, 0] < 1e-8:
            logging.warning('too small new range')
            limits[i, 0] -= 0.001
            limits[i, 1] += 0.001
    print('new limits = ', limits)
    return limits


def main(args):

    # Initialization
    # if len(args) > 1:
    #     input_path = args[1]
    # else:
    #     # input_path = os.path.join('../runs_abc/', 'params.yml')
    #     input_path = os.path.join('../runs_abc/', 'params.yml')

    # input = yaml.load(open(input_path, 'r'))

    ### Paths
    # path = input['path']
    # path = {'output': os.path.join('../runs_abc/', 'output/'), 'valid_data': '../rans_ode/valid_data/'}
    path = {'output': os.path.join('../../', 'rans_output_nominal/'), 'valid_data': '../rans_ode/valid_data/'}
    # path = {'output': os.path.join('../les_closure/ABC/', 'output/'), 'valid_data': '../les_closure/valid_data/'}

    print(path)
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(path['output'], 'ABClog_postprocess0005')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('\n############# POSTPROCESSING ############')
    x_list = [0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0005]
    num_bin_kde = 100
    num_bin_raw = 60
    C_limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
    N_params = len(C_limits)
    files_abc = glob.glob1(path['output'], "classic_abc*.npz")
    files = [os.path.join(path['output'], i) for i in files_abc]

    logging.info('Loading data')
    accepted = load_c(files, N_params)
    dist = load_dist(files)
    ####################################################################################################################
    #
    # # ################################################################################################################
    logging.info('\n############# Classic ABC ############')
    output_folder = os.path.join(path['output'], 'postprocess')
    output_by_percent(accepted, dist, C_limits, x_list, num_bin_raw, num_bin_kde, output_folder, mirror=False)



if __name__ == '__main__':
    main(sys.argv)