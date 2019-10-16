import numpy as np
import logging
import os
import sys
sys.path.append('/Users/olgadorr/Research/ABC_MCMC')
from time import time

from work_func_overflow import work_function
from overflow_driver import Overflow
import sumstat
from pyabc.utils import timer
import pyabc.glob_var as g


def calibration_loop(overflow, job_folder, c_array):

    start = time()

    for i, c in enumerate(c_array):
        with open(os.path.join(job_folder, 'result.dat'), 'a+') as f:
            logging.info('{} {}'.format(i, c))
            result = work_function(overflow, c, i)
            f.write('{}\n'.format(result))
    end = time()
    timer(start, end, 'Time ')
    return


def main():

    n_job = int(sys.argv[1])
    restart = 0
    if len(sys.argv) > 2:
        restart = 1

    N_proc = 1

    basefolder = './'
    output_folder = '../output'
    data_folder = os.path.join(basefolder, 'valid_data', )
    job_folder = os.path.join(output_folder, 'calibration_job{}'.format(n_job))
    exe_dir = '/nobackup/odoronin/over2.2n/'

    logPath = job_folder
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'calibration')), logging.StreamHandler()],
        level=logging.DEBUG)
    ################################################################################
    c_array = np.loadtxt(os.path.join(job_folder, 'c_array_{}'.format(n_job)))
    if restart:
        done = len(np.load(os.path.join(job_folder, 'results.dat')))
        c_array = c_array[done:]

    overflow = Overflow(job_folder, data_folder, exe_dir, N_proc)
    g.job_folder = job_folder
    g.Grid = sumstat.GridData(data_folder)
    g.Truth = sumstat.TruthData(data_folder, ['cp', 'u', 'uv'])
    calibration_loop(overflow, job_folder, c_array)


if __name__ == '__main__':
    main()
