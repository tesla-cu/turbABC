import os
import sys
import logging
import yaml
import numpy as np
sys.path.append('/Users/olgadorr/Research/ABC_MCMC')
print(sys.path)
import sumstat
import model
import data
import workfunc_les

import pyabc.glob_var as g
import pyabc.parallel as parallel
import pyabc.abc_alg as abc_alg
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

    les_data = data.DataFiltered(valid_folder=g.path['valid_data'], case_params=input['data_params'])
    g.Truth = sumstat.TruthData(valid_folder=g.path['valid_data'], data_params=input['data_params'],
                                sumstat_params=input['sumstat_params'])
    g.SumStat = sumstat.SummaryStatistics(sumstat_params=input['sumstat_params'])
    g.LesModel = model.NonlinearModel(data=les_data, model_params=input['model'],
                                      n_data_points=input['data_params']['N_point'],
                                      calc_strain_flag=g.SumStat.sumstat_type != 'sigma_pdf_log')

    C_limits = np.array(input['C_limits'])[:input['model']['N_params']]
    logging.info('C_limits: {}'.format(C_limits))
    np.savetxt(os.path.join(g.path['output'], 'C_limits_init'), C_limits)

    g.work_function = workfunc_les.abc_work_function
    g.par_process = parallel.Parallel(1, input['parallel_threads'])
    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    if input['abc_algorithm'] == 'abc':  # classical abc algorithm (accepts all samples for further postprocessing)
        C_array = abc_alg.sampling(algorithm_input['sampling'], C_limits, algorithm_input['N'])
        abc_alg.abc_classic(C_array)
        ################################################################################################################
    elif input['abc_algorithm'] == 'abc_IMCMC':  # MCMC with calibration step (Wegmann 2009)
        logging.info("ABC-MCMC algorithm")
        logging.info('Calibration')
        abc_alg.two_calibrations(algorithm_input, C_limits)
        logging.info('Chains')
        g.N_per_chain = algorithm_input['N_per_chain']
        g.t0 = algorithm_input['t0']
        abc_alg.mcmc_chains(n_chains=g.par_process.proc)
        ################################################################################################################
    elif input['abc_algorithm'] == 'abc_MCMC_adaptive':
        logging.info("ABC-MCMC algorithm with adaptation")
        logging.info('Chains')
        g.C_limits = C_limits
        g.N_per_chain = algorithm_input['N_per_chain']
        g.target_acceptance = algorithm_input['target_acceptance']
        g.t0 = algorithm_input['t0']
        C_start = abc_alg.sampling('random', C_limits, input['parallel_threads'])
        np.savetxt(os.path.join(g.path['calibration'], 'C_start'), C_start)
        abc_alg.mcmc_chains(n_chains=g.par_process.proc)
    else:
        logging.warning('{} algorithm does not exist'.format(input['abc_algorithm']))
    ####################################################################################################################


if __name__ == '__main__':
    main()
