import numpy as np
import os
from odeSolveMethods import RungeKuttaFehlberg as RK
import plotting
import utils
import logging
from time import time
import postprocess
import parallel
import abc_alg
import glob_var as g

g.path = {'output': './output/', 'plots': './plots/'}

if not os.path.isdir(g.path['output']):
    os.makedirs(g.path['output'])


logPath = g.path['output']
logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'ABC_log')), logging.StreamHandler()],
        level=logging.DEBUG)


def main():

    g.par_process = parallel.Parallel(1, g.N_proc)
    if g.algorithm == 'abc':
        logging.info('C_limits:{}'.format(g.C_limits))
        np.savetxt(os.path.join(g.path['output'], 'C_limits'), g.C_limits)
        C_array = abc_alg.sampling('uniform', g.C_limits, g.N)
        calibration, dist = abc_alg.main_loop(abc_alg.abc_work_function_periodic, C_array)
        np.savez(os.path.join(g.path['output'], 'calibration.npz'), C=calibration, dist=dist)
        ################################################################################################################
    else:
        calibration, dist = abc_alg.main_loop_IMCMC(work_func=abc_alg.abc_work_function_periodic)
        np.savez(os.path.join(g.path['output'], 'calibration.npz'), C=calibration, dist=dist)
    ########################################################################################################################
    # eps_k = 0.06
    # dist_k = dist
    # accepted = calibration[np.where(dist_k.max(1) < eps_k)[0]]
    # logging.info('accepted.shape: {}'.format(accepted.shape))
    # if accepted.shape[0] == 0:
    #     logging.info("There is no accepted parametes, consider increasing eps.")
    #     exit()
    # np.savez(os.path.join(path['output'], 'accepted.npz'), C=accepted)
    # ########################################################################################################################
    # num_bin_joint = 20
    # Z, C_final_smooth = postprocess.calc_final_C(accepted, num_bin_joint, C_limits, path)
    # postprocess.calc_marginal_pdf(Z, num_bin_joint, C_limits, path)


if __name__ == '__main__':
    main()