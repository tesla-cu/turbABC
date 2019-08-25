import os
import sys
import logging
import yaml
import numpy as np
sys.path.append('/Users/olgadorr/Research/ABC_RANS')
import pyabc.parallel as parallel
import pyabc.abc_alg as abc_alg
import pyabc.glob_var as g
import sumstat
from workfunc_rans import StrainTensor


def main():

    # Initialization
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join('./', 'rans_ode.yml')
    input = yaml.load(open(input_path, 'r'))

    ### Paths
    g.path = input['path']
    if not os.path.isdir(g.path['output']):
        os.makedirs(g.path['output'])
    g.path['calibration'] = os.path.join(g.path['output'], 'calibration')
    if not os.path.isdir(g.path['calibration']):
        os.makedirs(g.path['calibration'])

    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(g.path['output'], 'ABC_log')), logging.StreamHandler()],
        level=logging.DEBUG)

    # ABC algorithm
    algorithm_input = input['algorithm'][input['abc_algorithm']]
    # RANS ode specific
    g.Truth = sumstat.TruthData(valid_folder=g.path['valid_data'], case=input['case'])
    g.Strain = StrainTensor(valid_folder=g.path['valid_data'])
    g.case = input['case']
    C_limits = np.array(input['C_limits'])
    ####################################################################################
    # Run
    ####################################################################################
    g.par_process = parallel.Parallel(1, input['parallel_threads'])
    logging.info('C_limits: [{}, {}, {}, {}]'.format(C_limits[0], C_limits[1], C_limits[2], C_limits[3]))
    np.savetxt(os.path.join(g.path['output'], 'C_limits_init'), C_limits)
    if input['abc_algorithm'] == 'abc':    # classical abc algorithm (accepts all samples for further postprocessing)
        logging.info("Classic ABC algorithm")
        C_array = abc_alg.sampling(algorithm_input['sampling'], C_limits, algorithm_input['N'])
        abc_alg.abc_classic(C_array)
        ################################################################################################################
    elif input['abc_algorithm'] == 'abc_IMCMC':    # MCMC with calibration step (Wegmann 2009)
        logging.info("ABC-MCMC algorithm")
        logging.info('Calibration')
        C_array_for_chains = abc_alg.calibration(algorithm_input, C_limits)
        logging.info('Chains')
        g.N_per_chain = algorithm_input['N_per_chain']
        g.t0 = algorithm_input['t0']
        abc_alg.mcmc_chains(C_array_for_chains)
    elif input['abc_algorithm'] == 'abc_MCMC_adaptive':
        logging.info("ABC-MCMC algorithm with adaptation")
        logging.info('Chains')
        g.C_limits = C_limits
        g.N_per_chain = algorithm_input['N_per_chain']
        g.target_acceptance = algorithm_input['target_acceptance']
        g.t0 = algorithm_input['t0']
        C_array = abc_alg.sampling('random', C_limits, input['parallel_threads'])
        abc_alg.mcmc_chains(C_array, adaptive=True)
    else:
        logging.warning('{} algorithm does not exist'.format(input['abc_algorithm']))
    ####################################################################################################################
    # postprocess.main()


if __name__ == '__main__':
    main()
