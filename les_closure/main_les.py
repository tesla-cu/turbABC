import os
import sys
import logging
import yaml
import numpy as np
sys.path.append('/Users/olgadorr/Research/ABC_MCMC')
print(sys.path)
import sumstat
import pyabc.glob_var as g
# import pyabc.data as data
# import pyabc.model as model
# import pyabc.parallel as parallel
#


def main():

    # # Initialization
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join('./', 'params.yml')
    input = yaml.load(open(input_path, 'r'))

    ### Paths
    g.path = input['path']
    print('path', g.path)
    if not os.path.isdir(g.path['output']):
        os.makedirs(g.path['output'])
    if input['abc_algorithm'] == 'abc_IMCMC':
        g.path['calibration'] = os.path.join(g.path['output'], 'calibration')
        if not os.path.isdir(g.path['calibration']):
            os.makedirs(g.path['calibration'])
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(g.path['output'], 'ABC_log')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))
    ####################################################################################################################
    # Preprocess
    ####################################################################################################################
    print(input['sumstat_params']['sumstat'])
    algorithm_input = input['algorithm'][input['abc_algorithm']]
    g.Truth = sumstat.TruthData(valid_folder=g.path['valid_data'],
                              data_params=input['data_params'], sumstat_params=input['sumstat_params'])
    print(g.Truth)
    if input['data']['load'] == 0:
        init.create_LES_TEST_data(params.data, params.physical_case, params.compare_pdf)
        g.TEST_sp = data.DataSparse(params.data['data_path'], 0, g.TEST, params.abc['num_training_points'])
        g.LES = None
        g.TEST = None
        path = os.path.join(params.data['data_path'], 'T.npz')
        g.TEST_Model = model.NonlinearModel(path, 0, params.model, params.abc, params.algorithm, params.C_limits,
                                            params.compare_pdf, params.abc['random'], g.TEST_sp)
    elif params.data['load'] == 1:
        init.load_LES_TEST_data(params.data, params.physical_case, params.compare_pdf)
        g.TEST_sp = data.DataSparse(params.data['data_path'], 0, g.TEST, params.abc['num_training_points'])
        g.LES = None
        g.TEST = None
        path = os.path.join(params.data['data_path'], 'T.npz')
        g.TEST_Model = model.NonlinearModel(path, 0, params.model, params.abc, params.algorithm, params.C_limits,
                                            params.compare_pdf, params.abc['random'], g.TEST_sp)
    elif params.data['load'] == 2:
        g.TEST_sp = data.DataSparse(params.data['data_path'], 1)
        path = os.path.join(params.data['data_path'], 'T.npz')
        g.TEST_Model = model.NonlinearModel(path, 0, params.model, params.abc, params.algorithm, params.C_limits,
                                            params.compare_pdf, params.abc['random'], g.TEST_sp)
    elif params.data['load'] == 3:
        path = os.path.join(params.data['data_path'], 'T.npz')
        g.TEST_Model = model.NonlinearModel(path, 1, params.model, params.abc, params.algorithm, params.C_limits,
                                            params.compare_pdf, params.abc['random'])
        g.sum_stat_true = np.load(os.path.join(params.data['data_path'], 'sum_stat_true.npz'))
    #
    # if params.parallel['N_proc'] > 1:
    #     g.par_process = parallel.Parallel(params.parallel['progressbar'], params.parallel['N_proc'])
    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    # abc = abc_class.ABC(params.abc, params.algorithm, params.model['N_params'], params.parallel['N_proc'],
    #                     params.C_limits)
    # abc.main_loop()
    # comm.Barrier()
    # np.savez(os.path.join(g.path['output'], 'accepted_{}.npz'.format(rank)), C=g.accepted, dist=g.dist)
    #
    # logging.info('Accepted parameters and distances saved in {}'.format(os.path.join(g.path['output'],
    #                                                                                  'accepted_{}.npz'.format(rank))))


if __name__ == '__main__':
    main()
