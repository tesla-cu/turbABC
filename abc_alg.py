import numpy as np
import logging
import os
from time import time
import utils
import glob_var as g
from odeSolveMethods import RungeKuttaFehlberg as RK
from odeSolveMethods import BDF
# Load in validation data
folder = './valid_data/'
axi_exp_k = np.loadtxt(os.path.join(folder, 'axi_exp_k.txt'))
axi_exp_b = np.loadtxt(os.path.join(folder, 'axi_exp_b.txt'))
axi_con_k = np.loadtxt(os.path.join(folder, 'axi_con_k.txt'))
axi_con_b = np.loadtxt(os.path.join(folder, 'axi_con_b.txt'))
shear_k = np.loadtxt(os.path.join(folder, 'shear_k.txt'))
plane_k = np.loadtxt(os.path.join(folder, 'plane_k.txt'))
plane_b = np.loadtxt(os.path.join(folder, 'plane_b.txt'))
period1_k = np.loadtxt(os.path.join(folder, 'period1_k.txt'))
period2_k = np.loadtxt(os.path.join(folder, 'period2_k.txt'))
period3_k = np.loadtxt(os.path.join(folder, 'period3_k.txt'))
period4_k = np.loadtxt(os.path.join(folder, 'period4_k.txt'))
period5_k = np.loadtxt(os.path.join(folder, 'period5_k.txt'))

S_axi_exp = utils.axisymmetric_expansion()
S_axi_con = utils.axisymmetric_contraction()
S_pure_shear = utils.pure_shear()
S_plane_strain = utils.plane_strain()
S_periodic = np.zeros(6)



def calc_err(x, y, valid_data_x, valid_data_y):
    points = np.interp(valid_data_x, x, y)
    points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))   # adding gaussian noise
    diff = points - valid_data_y
    return np.max(np.abs(diff))


def rans(t, x, args):
    """
    :param x:
    :return:
    """
    c, S = args

    k = x[0]                # turbulence kinetic energy
    e = x[1]                # dissipation rate
    a = np.array(x[2:])     # anisotropy tensor

    P = -k * (np.sum(a * S) + np.sum(a[3:] * S[3:]))

    alf1 = P / e - 1 + c[0]
    alf2 = c[1] - 4 / 3

    # Governing equations
    dx = np.zeros(8)
    dx[0] = P - e                               # dk / dt
    dx[1] = (c[2] * P - c[3] * e) * e / k         # de / dt
    dx[2:] = -alf1 * e * a / k + alf2 * S       # d a_ij / dt
    return dx


def rans_periodic(t, x, args):
    """

    :param x:
    :return:
    """
    c, s0, beta = args

    S = S_periodic
    S[3] = (s0 / 2) * np.sin(beta * s0 * t)  # applied periodic shear

    k = x[0]                # turbulence kinetic energy
    e = x[1]                # dissipation rate
    a = np.array(x[2:])     # anisotropy tensor

    P = -k * (np.sum(a * S) + np.sum(a[3:] * S[3:]))
    alf1 = P / e - 1 + c[0]
    alf2 = c[1] - 4 / 3
    # Governing equations
    dx = np.zeros(8)
    dx[0] = P - e                               # dk / dt
    dx[1] = (c[2] * P - c[3] * e) * e / k         # de / dt
    dx[2:] = -alf1 * e * a / k + alf2 * S       # d a_ij / dt
    return dx


def sampling(sampling, C_limits, N):
    if sampling == 'random':
        array = utils.sampling_random(N, C_limits)
    elif sampling == 'uniform':
        array = utils.sampling_uniform_grid(N, C_limits)
    logging.info('Sampling is {}'.format(sampling))
    return array


def abc_work_function_impulsive(c):

    err = np.zeros(4)

    # axisymmetric expansion
    tspan = [0, 1.6 / np.abs(S_axi_exp[0])]
    Tnke, Ynke = RK(f=rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.05, args=(c, S_axi_exp))
    err[0] = calc_err(np.abs(S_axi_exp[0]) * Tnke, Ynke[:, 0], axi_exp_k[:, 0], axi_exp_k[:, 1])
    # err[9] = calc_err(np.abs(S_axi_exp[0]) * Tnke, Ynke[:, 2], axi_exp_b[:, 0], 2 * axi_exp_b[:, 1])

    # axisymmetric contraction
    tspan = [0, 1.6 / np.abs(S_axi_con[0])]
    Tnke, Ynke = RK(f=rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.05, args=(c, S_axi_con))
    err[1] = calc_err(np.abs(S_axi_con[0]) * Tnke, Ynke[:, 0], axi_con_k[:, 0], axi_con_k[:, 1])
    # err[10] = calc_err(np.abs(S_axi_con[0]) * Tnke, Ynke[:, 2], axi_con_b[:, 0], 2 * axi_con_b[:, 1])

    # pure shear
    tspan = [0, 5.2/ (2*S_pure_shear[3])]
    Tnke, Ynke = RK(f=rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.05, args=(c, S_pure_shear))
    err[2] = calc_err(2 * S_pure_shear[3] * Tnke, Ynke[:, 0], shear_k[:, 0], shear_k[:, 1])

    # plane strain
    tspan = [0, 1.6/S_plane_strain[0]]
    Tnke, Ynke = RK(f=rans, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.05, args=(c, S_plane_strain))
    err[3] = calc_err(np.abs(S_plane_strain[0]) * Tnke, Ynke[:, 0], plane_k[:, 0], plane_k[:, 1])
    # err[7] = calc_err(np.abs(S_plane_strain[0]) * Tnke, Ynke[:, 2], plane_b[:, 0], 2 * plane_b[:, 1])

    result = np.hstack((c, np.max(err))).tolist()
    return result


def abc_work_function_periodic(c):

    s0 = 3.3
    beta = [0.125, 0.25, 0.5, 0.75, 1]
    err = np.zeros(5)

    # Periodic shear(five different frequencies)
    # logging.info('C = {}'.format(c))
    tspan = [0, 51/s0]
    Tnke, Ynke = RK(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[0]))
    err[0] = calc_err(s0 * Tnke, Ynke[:, 0], period1_k[:, 0], period1_k[:, 1])
    Tnke, Ynke = BDF(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[1]))
    err[1] = calc_err(s0 * Tnke, Ynke[:, 0], period2_k[:, 0], period2_k[:, 1])
    Tnke, Ynke = BDF(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[2]))
    err[2] = calc_err(s0 * Tnke, Ynke[:, 0], period3_k[:, 0], period3_k[:, 1])
    Tnke, Ynke = BDF(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[3]))
    err[3] = calc_err(s0 * Tnke, Ynke[:, 0], period4_k[:, 0], period4_k[:, 1])
    Tnke, Ynke = BDF(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, s0, beta[4]))
    err[4] = calc_err(s0 * Tnke, Ynke[:, 0], period5_k[:, 0], period5_k[:, 1])

    result = np.hstack((c, np.max(err))).tolist()
    return result


def main_loop(work_func, C_array):
    N_params = len(C_array[0])
    start = time()
    g.par_process.run(func=work_func, tasks=C_array)
    result = g.par_process.get_results()
    end = time()
    if g.algorithm == 'abc':
        accepted = np.array([C[:N_params] for C in result])
        dist = np.array([C[N_params:] for C in result])
    else:
        accepted = np.array([chunk[:N_params] for item in result for chunk in item])
        dist = np.array([chunk[-1] for item in result for chunk in item])
    utils.timer(start, end, 'Time ')
    logging.debug('Number of tested parameters: {}'.format(len(accepted)))
    return accepted, dist


def main_loop_IMCMC(work_func):

    C_array = sampling('uniform', g.C_limits, 5)
    logging.info('Calibration step with {} samples'.format(len(C_array)))

    start_calibration = time()
    g.par_process.run(func=work_func, tasks=C_array)
    S_init = g.par_process.get_results()
    end_calibration = time()
    utils.timer(start_calibration, end_calibration, 'Time of calibration step')

    # Define epsilon
    logging.info('x = {}'.format(g.x))
    S_init.sort(key=lambda y: y[-1])
    S_init = np.array(S_init)
    g.eps = np.percentile(S_init, q=int(g.x * 100), axis=0)[-1]
    logging.info('eps after calibration step = {}'.format(g.eps))

    # Define std
    phi = 1
    S_init = S_init[np.where(S_init[:, -1] < g.eps)]
    g.std = phi*np.std(S_init[:, :-1], axis=0)
    logging.info('std for each parameter after calibration step:\n{}'.format(g.std))

    # Define new range
    for i in range(len(g.C_limits)):
        max_S = np.max(S_init[:, i])
        min_S = np.min(S_init[:, i])
        half_length = phi * (max_S - min_S) / 2.0
        middle = (max_S + min_S) / 2.0
        g.C_limits[i] = np.array([middle - half_length, middle + half_length])
    logging.info('New parameters range after calibration step:\n{}'.format(g.C_limits))
    np.savetxt(os.path.join(g.path['output'], 'C_limits'), g.C_limits)

    # Randomly choose starting points for Markov chains
    C_start = (S_init[np.random.choice(S_init.shape[0], g.par_process.proc, replace=False), :-1])
    np.set_printoptions(precision=3)
    logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
    C_array = C_start.tolist()
    ####################################################################################################################
    # Markov chains
    return main_loop(work_function_MCMC, C_array)


def work_function_MCMC(C_init):
    N = g.N_chain
    C_limits = g.C_limits

    N_params = len(C_init)

    std = g.std
    eps = g.eps

    result = np.empty((N, N_params + 1), dtype=np.float32)
    s_d = 2.4 ** 2 / N_params  # correct covariance according dimensionality
    t0 = 50  # initial period witout adaptation

    # add first param

    result[0, :] = abc_work_function_impulsive(C_init)

    mean_prev = 0
    cov_prev = 0

    ####################################################################################################################
    def mcmc_loop(i, counter_sample, counter_dist):
        nonlocal mean_prev, cov_prev
        while True:
            while True:
                # print(i, counter_dist, counter_sample)
                counter_sample += 1
                if i < t0:
                    c = np.random.normal(result[i - 1, :-1], std)
                elif i == t0:
                    mean_prev = np.mean(result[:t0, :-1], axis=0)
                    cov_prev = s_d * np.cov(result[0:t0, :-1].T)
                    c = np.random.multivariate_normal(result[i - 1, :-1], cov=cov_prev)
                else:
                    cov_prev, mean_prev = utils.covariance_recursive(result[i - 1, :-1], i - 1, cov_prev, mean_prev, s_d)
                    c = np.random.multivariate_normal(result[i - 1, :-1], cov=cov_prev)
                if not (False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                    break
            distance = abc_work_function_impulsive(c)
            counter_dist += 1
            if distance[-1] <= eps:
                result[i, :] = distance
                break
        return counter_sample, counter_dist

    try:
        from tqdm import tqdm
        tqdm_flag = 1
    except ImportError:
        tqdm_flag = 0

    if tqdm_flag == 1:
        with tqdm(total=N) as pbar:
            pbar.update()
            # Markov Chain
            counter_sample = 0
            counter_dist = 0
            for i in range(1, N):
                counter_sample, counter_dist = mcmc_loop(i, counter_sample, counter_dist)
                pbar.update()
            pbar.close()
    else:
        # Markov Chain
        counter_sample = 0
        counter_dist = 0
        for i in range(1, N):
            counter_sample, counter_dist = mcmc_loop(i, counter_sample, counter_dist)
            if i % 1000 == 0:
                logging.info("Accepted {} samples".format(i))

    print('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    print('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    logging.info('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    logging.info('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    return result.tolist()