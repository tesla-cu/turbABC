import numpy as np
import os
import logging
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

    # Initialization
    # Run
    g.par_process = parallel.Parallel(1, g.N_proc)
    if g.algorithm == 'abc':
        logging.info('C_limits:{}'.format(g.C_limits))
        np.savetxt(os.path.join(g.path['output'], 'C_limits'), g.C_limits)
        C_array = abc_alg.sampling('uniform', g.C_limits, g.N)
        calibration, dist = abc_alg.main_loop(C_array)
        np.savez(os.path.join(g.path['output'], 'calibration.npz'), C=calibration, dist=dist)
        ################################################################################################################
    else:
        calibration, dist = abc_alg.main_loop_IMCMC()
        np.savez(os.path.join(g.path['output'], 'calibration.npz'), C=calibration, dist=dist)
    ####################################################################################################################
    postprocess.main()


if __name__ == '__main__':
    main()
