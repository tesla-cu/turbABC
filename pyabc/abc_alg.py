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
    all_samples = np.array([C[:N_params] for C in result])
    sumstat = np.array([C[N_params:-1] for C in result])
    dist = np.array([C[-1] for C in result])
    writing_size = 1e7
    if N > writing_size:
        n = int(N // writing_size)
        for i in range(n):
            np.savez(os.path.join(g.path['output'], 'classic_abc{}.npz'.format(i)),
                     C=all_samples[i*writing_size:(i+1)*writing_size], dist=dist[i*writing_size:(i+1)*writing_size])
        if N % writing_size != 0:
            np.savez(os.path.join(g.path['output'], 'classic_abc{}.npz'.format(n)),
                     C=all_samples[n * writing_size:], dist=dist[n * writing_size:])
    else:
        np.savez(os.path.join(g.path['output'], 'classic_abc0.npz'), C=all_samples, sumstat=sumstat, dist=dist)
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

    print('Number of inf = ', np.sum(np.isinf(np.array(S_init)[:, -1])))
    # Define epsilon
    logging.info('x = {}'.format(x[0]))
    S_init.sort(key=lambda y: y[-1])
    S_init = np.array(S_init)
    eps = np.percentile(S_init, q=int(x[0] * 100), axis=0)[-1]
    logging.info('eps after calibration step = {}'.format(eps))
    np.savetxt(os.path.join(g.path['calibration'], 'eps1'), [eps])
    np.savez(os.path.join(g.path['calibration'], 'calibration1.npz'),
             C=S_init[:, :N_params], sumstat=S_init[:, N_params:-1], dist=S_init[:, -1])
    # Define new range
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
    logging.info('x = {}'.format(x[1]))
    S_init.sort(key=lambda y: y[-1])
    S_init = np.array(S_init)
    g.eps = np.percentile(S_init, q=int(x[1] * 100), axis=0)[-1]
    logging.info('eps after calibration step = {}'.format(g.eps))
    np.savetxt(os.path.join(g.path['calibration'], 'eps2'), [g.eps])
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
    np.savetxt(os.path.join(g.path['calibration'], 'C_start'), C_start)
    return


def mcmc_chains(n_chains):

    # N_params = len(C_init[0])
    start = time()
    g.par_process.run(func=one_chain, tasks=np.arange(n_chains))
    result = g.par_process.get_results()
    end = time()
    # accepted = np.array([chunk[:N_params] for item in result for chunk in item])
    # sumstat = np.array([chunk[N_params:-1] for item in result for chunk in item])
    # dist = np.array([chunk[-1] for item in result for chunk in item])
    utils.timer(start, end, 'Time for running chains')
    # np.savez(os.path.join(g.path['output'], 'accepted.npz'), C=accepted, sumstat=sumstat, dist=dist)
    # logging.debug('Number of accepted parameters: {}'.format(len(accepted)))
    return


def one_chain(id):
    N = g.N_per_chain
    C_limits = g.C_limits
    N_params = len(g.C_limits)
    C_init = np.loadtxt(os.path.join(g.path['calibration'], 'C_start')).reshape((-1, N_params))[id]
    N_params = len(C_init)
    work_function = workfunc_rans.define_work_function()
    result = np.empty((N, N_params + len(g.Truth.sumstat_true) + 1), dtype=np.float32)
    s_d = 2.4 ** 2 / N_params  # correct covariance according to dimensionality

    # add first param
    result[0, :] = work_function(C_init)

    if g.target_acceptance is not None:
        delta = result[0, -1]
        std = np.sqrt(0.1 * (C_limits[:, 1] - C_limits[:, 0]))
        target_acceptance = g.target_acceptance
        
    ####################################################################################################################
    def mcmc_step_burn_in(i):
        nonlocal counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = np.random.normal(result[i - 1, :N_params], g.std)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            sample_value = work_function(c)
            counter_dist += 1
            if sample_value[-1] <= g.eps:       # distance < epsilon
                result[i, :] = sample_value
                break
        return

    def mcmc_step(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = np.random.multivariate_normal(result[i - 1, :N_params], cov=s_d*cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            distance = work_function(c)
            counter_dist += 1
            if distance[-1] <= g.eps:
                result[i, :] = distance
                break
        cov_prev, mean_prev = utils.covariance_recursive(result[i, :N_params], i, cov_prev, mean_prev)
        return

    ####################################################################################################################
    def mcmc_step_burn_in_prior(i):
        nonlocal counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = np.random.normal(result[i - 1, :N_params], g.std)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            sample_value = work_function(c)
            counter_dist += 1
            if sample_value[-1] <= g.eps:
                prior_values = g.prior_interpolator([result[i - 1, :N_params], c])
                if np.random.random() < prior_values[0]/prior_values[1]:
                    result[i, :] = sample_value
                    break
        return

    def mcmc_step_prior(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist
        while True:
            while True:
                counter_sample += 1
                c = np.random.multivariate_normal(result[i - 1, :N_params], cov=s_d*cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            sample_value = work_function(c)
            counter_dist += 1
            if sample_value[-1] <= g.eps:
                prior_values = g.prior_interpolator([result[i - 1, :N_params], c])
                if np.random.random() < prior_values[0] / prior_values[1]:
                    result[i, :] = sample_value
                    break
        cov_prev, mean_prev = utils.covariance_recursive(result[i, :N_params], i, cov_prev, mean_prev)
        return

    ####################################################################################################################
    def mcmc_step_burn_in_adaptive(i):
        nonlocal counter_sample, counter_dist, delta, std, target_acceptance
        while True:
            while True:
                counter_sample += 1
                c = np.random.normal(result[i - 1, :N_params], std)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            sample_value = work_function(c)
            counter_dist += 1
            if sample_value[-1] <= delta:  # distance < eps
                result[i, :] = sample_value
                delta *= np.exp((i + 1) ** (-2 / 3) * (target_acceptance - 1))
                break
            else:
                delta *= np.exp((i + 1) ** (-2 / 3) * target_acceptance)
        return

    def mcmc_step_adaptive(i):
        nonlocal mean_prev, cov_prev, counter_sample, counter_dist, delta
        while True:
            while True:
                counter_sample += 1
                c = np.random.multivariate_normal(result[i - 1, :N_params], cov=s_d * cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            sample_value = work_function(c)
            counter_dist += 1
            if sample_value[-1] <= delta:  # distance < eps
                result[i, :] = sample_value
                delta *= np.exp((i + 1) ** (-2 / 3) * (target_acceptance - 1))
                break
            else:
                delta *= np.exp((i + 1) ** (-2 / 3) * target_acceptance)
        cov_prev, mean_prev = utils.covariance_recursive(result[i, :N_params], i, cov_prev, mean_prev)
    return
    #######################################################
    # Markov Chain
    counter_sample = 0
    counter_dist = 0
    mean_prev = 0
    cov_prev = 0
    # if changed prior after calibration step
    if g.prior_interpolator is not None:
        mcmc_step_burn_in = mcmc_step_burn_in_prior
        mcmc_step = mcmc_step_prior
    elif g.target_acceptance is not None:
        mcmc_step_burn_in = mcmc_step_burn_in_adaptive
        mcmc_step = mcmc_step_adaptive
    else:
        mcmc_step_burn_in = mcmc_step_burn_in
        mcmc_step = mcmc_step

    # burn in period with constant variance
    for i in range(1, min(g.t0, N)):
        mcmc_step_burn_in(i)
    # define mean and covariance from burn-in period
    mean_prev = np.mean(result[:g.t0, :N_params], axis=0)
    cov_prev = s_d * np.cov(result[:g.t0, : N_params].T)
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
    np.savez(os.path.join(g.path['output'], 'chain{}.npz'),
             C=result[:, :N_params], sumstat=result[:, N_params:-1], dist=result[:, -1])
    return result.tolist()


