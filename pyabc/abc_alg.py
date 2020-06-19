import logging
import os
import numpy as np
from time import time
from scipy.interpolate import RegularGridInterpolator

import pyabc.utils as utils
import pyabc.kde as kde
import pyabc.glob_var as g


def sampling(sampling, C_limits, N):

    if sampling == 'random':
        logging.info(f'Sampling is uniformly random with {N} samples in {len(C_limits)}D parameter space')
        array = utils.sampling_random(N, C_limits)
    elif sampling == 'uniform':
        logging.info(f'Sampling is uniform grid with {N**len(C_limits)} samples in {len(C_limits)}D parameter space')
        array = utils.sampling_uniform_grid(N, C_limits)
    return array


def abc_classic(C_array):

    N_params = len(C_array[0])
    N = len(C_array)
    logging.info(f'Classic ABC algorithm: Number of parameters: {N_params}, Number of samples: {N}')
    start = time()
    g.par_process.run(func=g.work_function, tasks=C_array)
    result = g.par_process.get_results()
    end = time()
    utils.timer(start, end, 'Time ')
    c = np.array([C[:N_params] for C in result])
    sumstat = np.array([C[N_params:-1] for C in result])
    dist = np.array([C[-1] for C in result])
    n, r, size = utils.check_output_size(N, N_params, len(sumstat[0]))
    for i in range(n):
        np.savez(os.path.join(g.path['output'], 'classic_abc{}.npz'.format(i)),
                 C=c[i*size:(i+1)*size], sumstat=sumstat[i*size:(i+1)*size], dist=dist[i*size:(i+1)*size])
    if r:
        np.savez(os.path.join(g.path['output'], 'classic_abc{}.npz'.format(n)),
                 C=c[n * size:], sumstat=sumstat[n * size:], dist=dist[n * size:])
    return


###########################################################
#   CALIBRATION
###########################################################
def two_calibrations(algorithm_input, C_limits):

    S_init = calibration_loop(algorithm_input['sampling'], C_limits, algorithm_input['N_calibration'][0])
    C_limits = calibration_postprocess1(S_init, algorithm_input['x'][0], algorithm_input['phi'], C_limits)
    S_init = calibration_loop(algorithm_input['sampling'], C_limits, algorithm_input['N_calibration'][1])
    calibration_postprocess2(S_init, algorithm_input['x'][1], algorithm_input['phi'], C_limits)
    print(algorithm_input['prior_update'])
    if algorithm_input['prior_update']:
        update_prior(S_init, C_limits, algorithm_input['prior_update'])


def calibration_loop(sampling_type, C_limits, N_calibration):

    logging.info('Sampling {}'.format(sampling))
    C_array = sampling(sampling_type, C_limits, N_calibration)
    logging.info(f'Calibration step 1')
    start_calibration = time()
    g.par_process.run(func=g.work_function, tasks=C_array)
    S_init = g.par_process.get_results()
    end_calibration = time()
    utils.timer(start_calibration, end_calibration, 'Time of calibration step 1')
    logging.debug('After Calibration: Number of inf = ', np.sum(np.isinf(np.array(S_init)[:, -1])))
    return S_init


def calibration_postprocess1(S_init, x, phi, C_limits, output_folder=None):
    N_params = len(C_limits)
    if not output_folder:
        output_folder = g.path['calibration']
    # Define epsilon
    eps = utils.define_eps(S_init, x)
    print(eps)
    S_init = np.array(S_init)
    logging.info('eps after calibration1 step = {}'.format(eps))
    np.savetxt(os.path.join(output_folder, 'eps1'), [eps])
    np.savez(os.path.join(output_folder, 'calibration1.npz'),
             C=S_init[:, :N_params], sumstat=S_init[:, N_params:-1], dist=S_init[:, -1])

    g.C_limits = np.empty_like(C_limits)
    for i in range(N_params):
        max_S = np.max(S_init[:, i])
        min_S = np.min(S_init[:, i])
        half_length = phi * (max_S - min_S) / 2.0
        middle = (max_S + min_S) / 2.0
        g.C_limits[i] = np.array([middle - half_length, middle + half_length])
    logging.info('New parameters range after calibration step:\n{}'.format(g.C_limits))
    np.savetxt(os.path.join(output_folder, 'C_limits'), g.C_limits)
    return g.C_limits


def calibration_postprocess2(S_init, x, phi, C_limits):
    N_params = len(C_limits)
    # Define epsilon again
    g.eps = utils.define_eps(S_init, x)
    S_init = np.array(S_init)
    logging.info('eps after calibration2 step = {}'.format(g.eps))
    np.savetxt(os.path.join(g.path['calibration'], 'eps2'), [g.eps])
    np.savez(os.path.join(g.path['calibration'], 'calibration2.npz'),
             C=S_init[:, :N_params], sumstat=S_init[:, N_params:-1], dist=S_init[:, -1])

    # Define std
    S_init = S_init[np.where(S_init[:, -1] < g.eps)]
    g.std = phi*np.std(S_init[:, :N_params], axis=0)
    logging.info('std for each parameter after calibration step:{}'.format(g.std))
    np.savetxt(os.path.join(g.path['calibration'], 'std'), [g.std])
    for i, std in enumerate(g.std):
        if std < 1e-8:
            g.std += 1e-5
            logging.warning('Artificially added std! Consider increasing number of samples for calibration step')
            logging.warning('new std for each parameter after calibration step:{}'.format(g.std))

    # Randomly choose starting points for Markov chains
    C_start = (S_init[np.random.choice(S_init.shape[0], g.par_process.proc, replace=False), :N_params])
    np.set_printoptions(precision=3)
    logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
    np.savetxt(os.path.join(g.path['output'], 'C_start'), C_start)

    return


def update_prior(S_init, C_limits, num_bin_update):
    # update prior based on accepted parameters in calibration
    N_params = len(C_limits)
    prior = kde.kdepy_fftkde(data=np.array(S_init)[:, :N_params], a=C_limits[:, 0], b=C_limits[:, 1],
                             num_bin_joint=num_bin_update)
    map_calibration = kde.find_MAP_kde(prior, C_limits[:, 0], C_limits[:, 1])
    logging.info('Estimated parameter after calibration step is {}'.format(map_calibration))
    np.savez(os.path.join(g.path['calibration'], 'prior.npz'), Z=prior)
    np.savetxt(os.path.join(g.path['calibration'], 'C_final_smooth'), map_calibration)
    prior_grid = np.empty((N_params, num_bin_update+1))
    for i, limits in enumerate(g.C_limits):
        prior_grid[i] = np.linspace(limits[0], limits[1], num_bin_update+1)
    g.prior_interpolator = RegularGridInterpolator(prior_grid, prior, bounds_error=False)
    return


###########################################################
#   CHAINS
###########################################################
def mcmc_chains(n_chains):

    start = time()
    g.par_process.run(func=one_chain, tasks=np.arange(n_chains))
    end = time()
    utils.timer(start, end, 'Time for running chains')

    # result = g.par_process.get_results()
    # accepted = np.array([chunk[:N_params] for item in result for chunk in item])
    # sumstat = np.array([chunk[N_params:-1] for item in result for chunk in item])
    # dist = np.array([chunk[-1] for item in result for chunk in item])
    return


def result_split(x, N_params):
    return x[:N_params], x[N_params:-1], x[-1]


def chain_kernel_const(center, cov):
    return np.random.normal(center, g.std)


def chain_kernel_adaptive(center, cov):
    return np.random.multivariate_normal(center, cov=cov)


def one_chain(chain_id):
    N = g.N_per_chain
    C_limits = g.C_limits
    N_params = len(g.C_limits)
    C_init = np.loadtxt(os.path.join(g.path['output'], 'C_start')).reshape((-1, N_params))[chain_id]
    result_c = np.empty((N, N_params))
    result_sumstat = np.empty((N, len(g.Truth.sumstat_true)))
    result_dist = np.empty(N)
    s_d = 2.4 ** 2 / N_params  # correct covariance according to dimensionality

    # add first param
    if not g.restart_chain[chain_id]:
        result_c[0], result_sumstat[0], result_dist[0] = result_split(g.work_function(C_init), N_params)
        counter_sample = 0
        counter_dist = 0
        mean_prev = 0
        cov_prev = 0

        # TODO: figure out what it is
        if g.target_acceptance is not None:
            delta = result_dist[0]
            g.std = np.sqrt(0.1 * (C_limits[:, 1] - C_limits[:, 0]))
            target_acceptance = g.target_acceptance

    ####################################################################################################################
    def mcmc_step(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = chain_kernel(result_c[i - 1], s_d*cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            result, *other = g.work_function(c)
            counter_dist += 1
            if result[-1] <= g.eps:  # distance < epsilon
                result_c[i], result_sumstat[i], result_dist[i] = result_split(result, N_params)
                break
        if i >= g.t0:
            cov_prev, mean_prev = utils.covariance_recursive(result_c[i], i, cov_prev, mean_prev)
        return other

    ####################################################################################################################
    def mcmc_step_prior(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = chain_kernel(result_c[i - 1], s_d*cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            result, *other = g.work_function(c)
            counter_dist += 1
            if result[-1] <= g.eps:
                prior_values = g.prior_interpolator([result_c[i - 1], c])
                if np.random.random() < prior_values[0] / prior_values[1]:
                    result_c[i], result_sumstat[i], result_dist[i] = result_split(result, N_params)
                    break
        if i >= g.t0:
            cov_prev, mean_prev = utils.covariance_recursive(result_c[i], i, cov_prev, mean_prev)
        return other

    ####################################################################################################################
    def mcmc_step_adaptive(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist, delta
        while True:
            while True:
                counter_sample += 1
                c = chain_kernel(result_c[i - 1], s_d * cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            result, *other = g.work_function(c)
            counter_dist += 1
            if result[-1] <= delta:  # distance < eps
                result_c[i], result_sumstat[i], result_dist[i] = result_split(result, N_params)
                delta *= np.exp((i + 1) ** (-2 / 3) * (target_acceptance - 1))
                break
            else:
                delta *= np.exp((i + 1) ** (-2 / 3) * target_acceptance)
        if i >= g.t0:
            cov_prev, mean_prev = utils.covariance_recursive(result_c[i], i, cov_prev, mean_prev)
        return other
    #######################################################
    # Markov Chain
    # if changed prior after calibration step
    if g.prior_interpolator is not None:
        mcmc_step = mcmc_step_prior
    elif g.target_acceptance is not None:
        mcmc_step = mcmc_step_adaptive
    else:
        mcmc_step = mcmc_step

    if not g.restart_chain[chain_id]:
        # burn in period with constant variance
        chain_kernel = chain_kernel_const
        for i in range(1, min(g.t0, N)):
            mcmc_step(i)
        # define mean and covariance from burn-in period
        mean_prev = np.mean(result_c[:g.t0], axis=0)
        cov_prev = s_d * np.cov(result_c[:g.t0].T)
    else:
        mean_prev = np.loadtxt(os.path.join(g.path['output'], 'mean'))
        cov_prev = np.loadtxt(os.path.join(g.path['output'], 'covariance'))

    # start period with adaptation
    chain_kernel = chain_kernel_adaptive
    for i in range(g.t0, N):
        mcmc_step(i)
        if g.save_chain_step:
            result = np.hstack((result_c[-1], result_sumstat[-1], result_dist[-1]))
            g.save_chain_step(result, cov_prev, mean_prev, i, counter_sample, counter_dist, other)
        if i % int(N/100) == 0:
            logging.info("Accepted {} samples".format(i))
    #######################################################
    print('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    print('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    logging.info('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    logging.info('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    n, r, size = utils.check_output_size(N, N_params, len([0]) - N_params - 1)
    for i in range(n):
        np.savez(os.path.join(g.path['output'], 'chain{}_{}.npz'.format(chain_id, i)), C=result_c[i*size:(i+1)*size],
                 sumstat=result_sumstat[i*size:(i+1)*size], dist=result_dist[i*size:(i+1)*size])
    if r:
        np.savez(os.path.join(g.path['output'], 'chain{}_{}.npz'.format(chain_id, n)),
                 C=result_c[n * size:], sumstat=result_sumstat[n * size:], dist=result_dist[n * size:])

    return


