import numpy as np
import os
from odeSolveMethods import RungeKuttaFehlberg as RK
import plotting
import utils
import logging
from time import time
import postprocess
import parallel

path = {'output': './output/', 'plots': './plots/'}

if not os.path.isdir(path['output']):
    os.makedirs(path['output'])


logPath = path['output']
logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'ABC_log')), logging.StreamHandler()],
        level=logging.DEBUG)



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
s0 = 3.3
beta = [0.125, 0.25, 0.5, 0.75, 1]


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
    S[3] = (s0 / 2) * np.sin(beta * s0 * t)  #applied shear

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


def abc_work_function_impulsive(c):

    err = np.zeros(4)

    # axisymmetric expansion
    tspan = [0, 1.6 / S_axi_exp[0]]
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

    result = np.concatenate((c, err)).tolist()
    return result


def abc_work_function_periodic(c):

    err = np.zeros(5)

    # Periodic shear(five different frequencies)
    tspan = np.array([0, 51])/s0
    Tnke, Ynke = RK(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.1, args=(c, s0, beta[0]))
    err[0] = calc_err(s0 * Tnke, Ynke[:, 0], period1_k[:, 0], period1_k[:, 1])
    Tnke, Ynke = RK(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.1, args=(c, s0, beta[1]))
    err[1] = calc_err(s0 * Tnke, Ynke[:, 0], period2_k[:, 0], period2_k[:, 1])
    Tnke, Ynke = RK(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.1, args=(c, s0, beta[2]))
    err[2] = calc_err(s0 * Tnke, Ynke[:, 0], period3_k[:, 0], period3_k[:, 1])
    Tnke, Ynke = RK(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.1, args=(c, s0, beta[3]))
    err[3] = calc_err(s0 * Tnke, Ynke[:, 0], period4_k[:, 0], period4_k[:, 1])
    Tnke, Ynke = RK(f=rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.1, args=(c, s0, beta[4]))
    err[4] = calc_err(s0 * Tnke, Ynke[:, 0], period5_k[:, 0], period5_k[:, 1])

    result = np.concatenate((c, err)).tolist()
    return result


def main():

    C_limits = np.array([[1.0, 1.8],
                         [0.6, 0.77],
                         [1.7, 2],
                         [1.85, 2.05]])
    N = 20

    logging.info('C_limits:{}'.format(C_limits))
    np.savetxt(os.path.join(path['output'], 'C_limits'), C_limits)
    ########################################################################################################################
    par_process = parallel.Parallel(1, 4)
    C_array = sampling('uniform', C_limits, N)
    calibration, dist = main_loop_MCMC(abc_work_function_periodic, C_array, par_process)
    np.savez(os.path.join(path['output'], 'calibration.npz'), C=calibration, dist=dist)
    ########################################################################################################################
    eps_k = 0.06
    dist_k = dist
    accepted = calibration[np.where(dist_k.max(1) < eps_k)[0]]
    logging.info('accepted.shape: {}'.format(accepted.shape))
    if accepted.shape[0] == 0:
        logging.info("There is no accepted parametes, consider increasing eps.")
        exit()
    np.savez(os.path.join(path['output'], 'accepted.npz'), C=accepted)
    ########################################################################################################################
    num_bin_joint = 20
    Z, C_final_smooth = postprocess.calc_final_C(accepted, num_bin_joint, C_limits, path)
    postprocess.calc_marginal_pdf(Z, num_bin_joint, C_limits, path)


if __name__ == '__main__':
    main()