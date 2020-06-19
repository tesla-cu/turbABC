import logging
import numpy as np
import os
from time import time

from sumstat import calc_sum_stat
import pyabc.glob_var as g
import pyabc.distance as dist


if g.norm_order == 1:
    calc_err = dist.calc_err_norm1
elif g.norm_order == 2:
    calc_err = dist.calc_err_norm2


def work_function(overflow, c, i):

    logging.debug('Calibration {}: {}'.format(i, c))
    overflow.write_inputfile(c)
    overflow.run_overflow(i)    # run overflow
    os.rename(os.path.join(g.job_folder, 'q.save'), os.path.join(g.job_folder, 'q.restart'))  # rename
    overflow.write_debug()
    overflow.run_overflow(i)    # run debug to get q.turb file
    cp, sum_stat_u, sum_stat_uv, u_slice, uv_slice, u_surface, x_shock = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.x_slices, g.Grid.y_slices)
    sum_stat_cp = calc_sum_stat(g.Grid.grid_x[::-1], cp[::-1], g.Truth.cp[:, 0])
    sum_stat = np.hstack((sum_stat_cp, sum_stat_u, sum_stat_uv, x_shock))
    err = calc_err(sum_stat[:-2]/g.Truth.norm[:-2], g.Truth.sumstat_true[:-2])
    result = np.hstack((c, sum_stat, err)).tolist()

    return result, cp, u_slice, uv_slice, u_surface


def work_function_chain(overflow, c, i):

    logging.debug(f'Chain step {i}: {c}')
    overflow.write_inputfile(c)
    overflow.run_overflow(i)    # run overflow
    os.rename(os.path.join(g.job_folder, 'q.save'), os.path.join(g.job_folder, 'q.restart'))  # rename
    overflow.write_debug()
    overflow.run_overflow(i)    # run debug to get q.turb file
    cp, sum_stat_u, sum_stat_uv, u_slice, uv_slice, u_surface, x_shock = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.x_slices, g.Grid.y_slices)
    sum_stat_cp = calc_sum_stat(g.Grid.grid_x[::-1], cp[::-1], g.Truth.cp[:, 0])
    sum_stat = np.hstack((sum_stat_cp, sum_stat_u, sum_stat_uv, x_shock))
    err = calc_err(sum_stat[:-2]/g.Truth.norm[:-2], g.Truth.sumstat_true[:-2])
    result = np.hstack((c, sum_stat, err)).tolist()

    return result, (cp, u_slice, uv_slice, u_surface)


def save_chain_step(result, cov, mean, i, counter_sample, counter_dist, other):

    with open(os.path.join(g.path['output'], 'result.dat'), 'a+') as result_file:
        result_file.write(f'{result}\n')
    with open(os.path.join(g.path['output'], 'covariance'), 'a+') as cov_file:
        cov_file.write(f'{cov.flatten()}\n')
    with open(os.path.join(g.path['output'], 'mean'), 'a+') as mean_file:
        mean_file.write(f'{mean}\n')
    with open(os.path.join(g.path['output'], 'counter'), 'a+') as counter_file:
        counter_file.write(f'{i} {counter_sample} {counter_dist}\n')

    cp, u, uv, u_surface = other
    with open(os.path.join(g.path['output'], 'cp_all.bin'), 'ab') as f:
        f.write(bytearray(cp))
    with open(os.path.join(g.path['output'], 'u_slice.bin'), 'ab') as f:
        f.write(bytearray(u))
    with open(os.path.join(g.path['output'], 'uv_slice.bin'), 'ab') as f:
        f.write(bytearray(uv))
    with open(os.path.join(g.path['output'], 'u_surface.bin'), 'ab') as f:
        f.write(bytearray(u_surface))
    return
