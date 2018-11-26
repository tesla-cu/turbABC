import numpy as np
import os
from odeSolveMethods import RungeKuttaFehlberg as RK
import plotting
import utils
import logging
from time import time
import postprocess
import parallel

# Load in validation data
folder = './valid_data/'
axi_exp_k = np.loadtxt(os.path.join(folder, 'axi_exp_k.txt'))
axi_exp_b = np.loadtxt(os.path.join(folder, 'axi_exp_b.txt'))
axi_con_k = np.loadtxt(os.path.join(folder, 'axi_con_k.txt'))
axi_con_b = np.loadtxt(os.path.join(folder, 'axi_con_b.txt'))
shear_k = np.loadtxt(os.path.join(folder, 'shear_k.txt'))
plane_k = np.loadtxt(os.path.join(folder, 'plane_k.txt'))
plane_b = np.loadtxt(os.path.join(folder, 'plane_b.txt'))


def calc_err(x, y, valid_data_x, valid_data_y):
    diff = np.interp(valid_data_x, x, y) - valid_data_y
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



def sampling(sampling, C_limits, N):
    if sampling == 'random':
        array = utils.sampling_random(N, C_limits)
    elif sampling == 'uniform':
        array = utils.sampling_uniform_grid(N, C_limits)
    logging.info('Sampling is {}'.format(sampling))
    return array


def main_loop_MCMC(work_func, C_array, par_process=None):
    N_params = len(C_array[0])
    start = time()
    if par_process:
        par_process.run(func=work_func, tasks=C_array)
        result = par_process.get_results()
        end = time()
        accepted = np.array([C[:N_params] for C in result])
        dist = np.array([C[N_params:] for C in result])
    else:
        result = []
        for c in C_array:
            result.append(work_func(c))
        end = time()
        accepted = np.array([C[:N_params] for C in result])
        dist = np.array([C[N_params:] for C in result])
    utils.timer(start, end, 'Time ')
    logging.debug('Number of accepted parameters: {}'.format(len(accepted)))
    return accepted, dist


def abc_work_function(c):
    # print('C: ', c)
    err = np.zeros((2*4))
    S = utils.axisymmetric_expansion()
    Tnke, Ynke = RK(f=rans, tspan=[0, 5], u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    err[0] = calc_err(np.abs(S[0]) * Tnke, Ynke[:, 0], axi_exp_k[:, 0], axi_exp_k[:, 1])
    err[4] = calc_err(np.abs(S[0]) * Tnke, Ynke[:, 2], axi_exp_b[:, 0], 2 * axi_exp_b[:, 1])

    S = utils.axisymmetric_contraction()
    Tnke, Ynke = RK(f=rans, tspan=[0, 2] / np.abs(S[0]), u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01 / np.abs(S[0]),
                    args=(c, S))
    err[1] = calc_err(np.abs(S[0]) * Tnke, Ynke[:, 0], axi_con_k[:, 0], axi_con_k[:, 1])
    err[5] = calc_err(np.abs(S[0]) * Tnke, Ynke[:, 2], axi_con_b[:, 0], 2 * axi_con_b[:, 1])

    S = utils.pure_shear()
    Tnke, Ynke = RK(f=rans, tspan=[0, 2], u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    err[2] = calc_err(2 * S[3] * Tnke, Ynke[:, 0], shear_k[:, 0], shear_k[:, 1])

    S = utils.plane_strain()
    Tnke, Ynke = RK(f=rans, tspan=[0, 4], u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    err[3] = calc_err(np.abs(S[0]) * Tnke, Ynke[:, 0], plane_k[:, 0], plane_k[:, 1])
    err[7] = calc_err(np.abs(S[0]) * Tnke, Ynke[:, 2], plane_b[:, 0], 2 * plane_b[:, 1])

    result = np.concatenate((c, err)).tolist()
    return result

def main():

    C_limits = np.array([[1.35, 1.65],
                         [0.7, 0.76],
                         [1.65, 2],
                         [1.85, 2.1]])
    N = 40
    path = {'output': './output/', 'plots': './plots/'}
    ########################################################################################################################
    par_process = parallel.Parallel(2, 4)
    C_array = sampling('uniform', C_limits, N)
    calibration, dist = main_loop_MCMC(abc_work_function, C_array, par_process)
    np.savez(os.path.join(path['output'], 'calibration.npz'), C=calibration, dist=dist)
    ########################################################################################################################
    eps_k = 0.06
    dist_k = dist[:, :4]
    accepted = calibration[np.where(dist_k.max(1) < eps_k)[0]]
    np.savez(os.path.join(path['output'], 'accepted.npz'), C=accepted)
    ########################################################################################################################
    num_bin_joint = 10
    Z, C_final_smooth = postprocess.calc_final_C(accepted, num_bin_joint, C_limits, path)
    postprocess.calc_marginal_pdf(Z, num_bin_joint, C_limits, path)
    plotting.plot_marginal_smooth_pdf(path, C_limits)
    ########################################################################################################################

    # c = C_final_smooth[0]
    # print('C_final_smooth: ', c)
    # err = np.zeros(8)
    #
    # S = axisymmetric_expansion()
    # Tnke1, Ynke1 = RK(f=rans, tspan=[0, 5], u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    # err[0] = calc_err(np.abs(S[0]) * Tnke1, Ynke1[:, 0], axi_exp_k[:, 0], axi_exp_k[:, 1])
    # err[4] = calc_err(np.abs(S[0]) * Tnke1, Ynke1[:, 2], axi_exp_b[:, 0], 2 * axi_exp_b[:, 1])
    # x1, y1 = np.abs(S[0])*Tnke1, Ynke1[:, 0]
    #
    # S = axisymmetric_contraction()
    # Tnke2, Ynke2 = RK(f=rans, tspan=[0, 2] / np.abs(S[0]), u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01 / np.abs(S[0]),
    #             args=(c, S))
    # err[1] = calc_err(np.abs(S[0]) * Tnke2, Ynke2[:, 0], axi_con_k[:, 0], axi_con_k[:, 1])
    # err[5] = calc_err(np.abs(S[0]) * Tnke2, Ynke2[:, 2], axi_con_b[:, 0], 2 * axi_con_b[:, 1])
    # x2, y2 = np.abs(S[0])*Tnke2, Ynke2[:, 0]
    #
    # S = pure_shear()
    # Tnke3, Ynke3 = RK(f=rans, tspan=[0, 15], u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    # err[2] = calc_err(2 * S[3] * Tnke3, Ynke3[:, 0], shear_k[:, 0], shear_k[:, 1])
    # x3, y3 = 2*S[3]*Tnke3, Ynke3[:, 0]
    #
    # S = plane_strain()
    # Tnke4, Ynke4 = RK(f=rans, tspan=[0, 5], u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.01, args=(c, S))
    # err[3] = calc_err(np.abs(S[0]) * Tnke4, Ynke4[:, 0], plane_k[:, 0], plane_k[:, 1])
    # err[7] = calc_err(np.abs(S[0]) * Tnke4, Ynke4[:, 2], plane_b[:, 0], 2 * plane_b[:, 1])
    # x4, y4 =1/2*Tnke4, Ynke4[:, 0]
    #
    # print('err = ', err[:4])
    #
    # plotting.plot(x1, y1, x2, y2, x3, y3, x4, y4)

if __name__ == '__main__':
    main()