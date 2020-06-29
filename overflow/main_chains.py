import numpy as np
import logging
import os
import shutil
import sys
# sys.path.append('/nobackup/odoronin')
sys.path.append('/Users/olgadorr/Research/ABC_MCMC')
from time import time

from pyabc.utils import timer
import pyabc.glob_var as g
import pyabc.abc_alg as abc_alg

from overflow.work_func_overflow import work_function_chain, save_chain_step
from overflow.overflow_driver import Overflow
import overflow.sumstat as sumstat

N_PARAMS = 4


def main():

    n_job = sys.argv[1]
    output_folder = sys.argv[2]

    # N_proc = 16
    N_proc = 4
    # basefolder = '/nobackup/odoronin/overflow/'
    basefolder = './'
    # data_folder = os.path.join(basefolder, 'valid_data', )
    g.N_params = 4
    data_folder = '/Users/olgadorr/Research/ABC_MCMC/overflow/valid_data/'
    job_folder = os.path.join(output_folder, 'chain_{}'.format(n_job))
    exe_dir = os.path.join(basefolder, 'over2.2n', )

    g.path = {'output': job_folder}
    g.restart_chain = False
    g.save_chain_step = save_chain_step
    g.work_function = work_function_chain

    logPath = job_folder
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'chain')), logging.StreamHandler()],
        level=logging.DEBUG)
    ################################################################################
    c_init = np.loadtxt(os.path.join(job_folder, f'C_start'))
    done = 0
    if os.path.exists(os.path.join(job_folder, 'C_limits')):
        g.C_limits = np.loadtxt(os.path.join(job_folder, 'C_limits'))
    result_file = os.path.join(job_folder, 'result.dat')
    logging.info(result_file)
    logging.info(f"restart: {os.path.exists(result_file)}")
    if os.path.exists(result_file):
        with open(result_file) as f:
            lines = f.readlines()
            done = len(lines)
            last_result = np.fromstring(lines[-1][1:-1], dtype=float, sep=',')
            last_c = last_result[:N_PARAMS]
        logging.info(f'Restart from {done} step with c = {last_c}')
        c_init = last_c
        g.restart_chain = True
        g.t0 = 0
    np.savetxt(os.path.join(g.path['output'], f'C_start_{done}'), c_init)
    if not os.path.exists(os.path.join(job_folder, 'grid.in')):
        try:
            shutil.copy(os.path.join(data_folder, 'grid.in'), job_folder)
        except IOError as e:
            print("Unable to copy file. %s" % e)
        except:
            print("Unexpected error:", sys.exc_info())

    g.overflow = Overflow(job_folder, data_folder, exe_dir, N_proc)
    g.job_folder = job_folder
    g.Grid = sumstat.GridData(data_folder)
    g.Truth = sumstat.TruthData(data_folder, ['cp', 'u', 'uv', 'x_separation'])

    logging.info('Chains')
    g.N_per_chain = 1500  # long queue is 120 hours and approx. 1090 overflow runs
    g.t0 = 100
    abc_alg.one_chain(chain_id=0)


if __name__ == '__main__':
    main()