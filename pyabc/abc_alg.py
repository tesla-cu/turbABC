import logging
import os
import numpy as np
from time import time
from scipy.interpolate import RegularGridInterpolator

import pyabc.utils as utils
import pyabc.glob_var as g
import workfunc_rans


def sampling(sampling, C_limits, N):
    if sampling == 'random':
        array = utils.sampling_random(N, C_limits)
    elif sampling == 'uniform':
        array = utils.sampling_uniform_grid(N, C_limits)
    logging.info('Sampling is {}'.format(sampling))
    return array


def abc_classic(C_array):

    N_params = len(C_array[0])
    N = len(C_array)
    work_function = workfunc_rans.define_work_function()
    start = time()
    g.par_process.run(func=work_function, tasks=C_array)
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


def calibration(algorithm_input, C_limits):

    N_params = len(C_limits)
    work_function = workfunc_rans.define_work_function()
    x = algorithm_input['x']
    logging.info('Sampling {}'.format(algorithm_input['sampling']))
    C_array = sampling(algorithm_input['sampling'], C_limits, algorithm_input['N_calibration'][0])
    logging.info('Calibration step 1 with {} samples'.format(len(C_array)))

    start_calibration = time()
    g.par_process.run(func=work_function, tasks=C_array)
    S_init = g.par_process.get_results()
    end_calibration = time()
    utils.timer(start_calibration, end_calibration, 'Time of calibration step 1')
    logging.debug('After Calibration 1: Number of inf = ', np.sum(np.isinf(np.array(S_init)[:, -1])))

    # Define epsilon
    eps = utils.define_eps(S_init, x[0], N_params, id=1)
    logging.info('eps after calibration1 step = {}'.format(eps))
    np.savetxt(os.path.join(g.path['calibration'], 'eps1'), [eps])
    np.savez(os.path.join(g.path['calibration'], 'calibration1.npz'),
             C=S_init[:, :N_params], sumstat=S_init[:, N_params:-1], dist=S_init[:, -1])

    g.C_limits = np.empty_like(C_limits)
    for i in range(N_params):
        max_S = np.max(S_init[:, i])
        min_S = np.min(S_init[:, i])
        half_length = algorithm_input['phi'] * (max_S - min_S) / 2.0
        middle = (max_S + min_S) / 2.0
        g.C_limits[i] = np.array([middle - half_length, middle + half_length])
    logging.info('New parameters range after calibration step:\n{}'.format(g.C_limits))
    np.savetxt(os.path.join(g.path['calibration'], 'C_limits'), g.C_limits)

    # Second calibration step
    logging.info('Sampling {}'.format(algorithm_input['sampling']))
    C_array = sampling(algorithm_input['sampling'], g.C_limits, algorithm_input['N_calibration'][1])
    logging.info('Calibration step 2 with {} samples'.format(len(C_array)))
    start_calibration = time()
    g.par_process.run(func=work_function, tasks=C_array)
    S_init = g.par_process.get_results()
    end_calibration = time()
    utils.timer(start_calibration, end_calibration, 'Time of calibration step 2')

    # Define epsilon again
    eps = utils.define_eps(S_init, x[1])
    logging.info('eps after calibration2 step = {}'.format(eps))
    np.savetxt(os.path.join(g.path['calibration'], 'eps2'), [eps])
    np.savez(os.path.join(g.path['calibration'], 'calibration2.npz'),
             C=S_init[:, :N_params], sumstat=S_init[:, N_params:-1], dist=S_init[:, -1])

    # Define std
    S_init = S_init[np.where(S_init[:, -1] < g.eps)]
    g.std = algorithm_input['phi']*np.std(S_init[:, :N_params], axis=0)
    logging.info('std for each parameter after calibration step:{}'.format(g.std))
    np.savetxt(os.path.join(g.path['calibration'], 'std'), [g.std])
    for i, std in enumerate(g.std):
        if std < 1e-8:
            g.std += 1e-5
            logging.warning('Artificially added std! Consider increasing number of samples for calibration step')
            logging.warning('new std for each parameter after calibration step:{}'.format(g.std))

    # update prior based on accepted parameters in calibration
    if algorithm_input['prior_update']:
        prior, C_calibration = utils.gaussian_kde_scipy(data=S_init[:, :N_params],
                                                        a=g.C_limits[:, 0],
                                                        b=g.C_limits[:, 1],
                                                        num_bin_joint=algorithm_input['prior_update'])
        logging.info('Estimated parameter after calibration step is {}'.format(C_calibration))
        np.savez(os.path.join(g.path['calibration'], 'prior.npz'), Z=prior)
        np.savetxt(os.path.join(g.path['calibration'], 'C_final_smooth'), C_calibration)
        prior_grid = np.empty((N_params, algorithm_input['prior_update']+1))
        for i, limits in enumerate(g.C_limits):
            prior_grid[i] = np.linspace(limits[0], limits[1], algorithm_input['prior_update']+1)
        g.prior_interpolator = RegularGridInterpolator(prior_grid, prior, bounds_error=False)

    # Randomly choose starting points for Markov chains
    C_start = (S_init[np.random.choice(S_init.shape[0], g.par_process.proc, replace=False), :N_params])
    np.set_printoptions(precision=3)
    logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
    np.savetxt(os.path.join(g.path['output'], 'C_start'), C_start)
    return


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
    N_params = len(C_init)
    work_function = workfunc_rans.define_work_function()
    result_c = np.empty((N, N_params))
    result_sumstat = np.empty((N, len(g.Truth.sumstat_true)))
    result_dist = np.empty((N))
    s_d = 2.4 ** 2 / N_params  # correct covariance according to dimensionality

    # add first param
    result_c[0], result_sumstat[0], result_dist[0] = result_split(work_function(C_init), N_params)
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
            result = work_function(c)
            counter_dist += 1
            if result[-1] <= g.eps:  # distance < epsilon
                result_c[i], result_sumstat[i], result_dist[i] = result_split(result, N_params)
                break
        if i >= g.t0:
            cov_prev, mean_prev = utils.covariance_recursive(result_c[i], i, cov_prev, mean_prev)
        return

    ####################################################################################################################
    def mcmc_step_prior(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = chain_kernel(result_c[i - 1], s_d*cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            result = work_function(c)
            counter_dist += 1
            if result[-1] <= g.eps:
                prior_values = g.prior_interpolator([result_c[i - 1], c])
                if np.random.random() < prior_values[0] / prior_values[1]:
                    result_c[i], result_sumstat[i], result_dist[i] = result_split(result, N_params)
                    break
        if i >= g.t0:
            cov_prev, mean_prev = utils.covariance_recursive(result_c[i], i, cov_prev, mean_prev)
        return

    ####################################################################################################################
    def mcmc_step_adaptive(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist, delta
        while True:
            while True:
                counter_sample += 1
                c = chain_kernel(result_c[i - 1], s_d * cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            result = work_function(c)
            counter_dist += 1
            if result[-1] <= delta:  # distance < eps
                result_c[i], result_sumstat[i], result_dist[i] = result_split(result, N_params)
                delta *= np.exp((i + 1) ** (-2 / 3) * (target_acceptance - 1))
                break
            else:
                delta *= np.exp((i + 1) ** (-2 / 3) * target_acceptance)
        if i >= g.t0:
            cov_prev, mean_prev = utils.covariance_recursive(result_c[i], i, cov_prev, mean_prev)
        return
    #######################################################
    # Markov Chain
    counter_sample = 0
    counter_dist = 0
    mean_prev = 0
    cov_prev = 0
    # if changed prior after calibration step
    if g.prior_interpolator is not None:
        mcmc_step = mcmc_step_prior
    elif g.target_acceptance is not None:
        mcmc_step = mcmc_step_adaptive
    else:
        mcmc_step = mcmc_step

    # burn in period with constant variance
    chain_kernel = chain_kernel_const
    for i in range(1, min(g.t0, N)):
        mcmc_step(i)
    # define mean and covariance from burn-in period
    mean_prev = np.mean(result_c[:g.t0], axis=0)
    cov_prev = s_d * np.cov(result_c[:g.t0].T)
    chain_kernel = chain_kernel_adaptive
    # start period with adaptation
    for i in range(g.t0, N):
        mcmc_step(i)
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


