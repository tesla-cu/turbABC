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
    logging.debug(f'{calc_err.__name__}')
    overflow.write_inputfile(c)
    overflow.run_overflow()    # run overflow
    os.rename(os.path.join(g.job_folder, 'q.save'), os.path.join(g.job_folder, 'q.restart'))  # rename
    overflow.write_debug()
    overflow.run_overflow()    # run debug to get q.turb file
    cp, sum_stat_u, sum_stat_uv, u_slice, uv_slice, u_surface, x_shock = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.x_slices, g.Grid.y_slices)
    sum_stat_cp = calc_sum_stat(g.Grid.grid_x[::-1], cp[::-1], g.Truth.cp[:, 0])
    sum_stat = np.hstack((sum_stat_cp, sum_stat_u, sum_stat_uv, x_shock))
    err = calc_err(sum_stat[:-2]/g.Truth.norm[:-2], g.Truth.sumstat_true[:-2])
    result = np.hstack((c, sum_stat, err)).tolist()

    return result, cp, u_slice, uv_slice, u_surface


def work_function_chain(c):

    logging.debug(f'Chain step: {c}')
    overflow = g.overflow
    overflow.write_inputfile(c)
    success = overflow.run_overflow()    # run overflow
    if success:
        os.rename(os.path.join(g.job_folder, 'q.save'), os.path.join(g.job_folder, 'q.restart'))  # rename
        overflow.write_debug()
        overflow.run_overflow()    # run debug to get q.turb file
        output_tuple = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.x_slices, g.Grid.y_slices)
        cp, sum_stat_u, sum_stat_uv, u_slice, uv_slice, u_surface, x_shock = output_tuple
        sum_stat_cp = calc_sum_stat(g.Grid.grid_x[::-1], cp[::-1], g.Truth.cp[:, 0])
        sum_stat = np.hstack((sum_stat_cp, sum_stat_u, sum_stat_uv, x_shock))
        dist_x = calc_err(sum_stat[-2:], g.Truth.sumstat_true[-2:])
        logging.debug(f'separation error = {dist_x}')
        if dist_x < 0.25:
            err = calc_err(sum_stat[:-2]/g.Truth.norm[:-2], g.Truth.sumstat_true[:-2])
        else:
            err = 1e8
        result = np.hstack((c, sum_stat, err)).tolist()
        other = (cp, u_slice, uv_slice, u_surface, x_shock)
    else:
        result = np.hstack((c, np.zeros(g.Truth.sumstat_true), 10e8)).tolist()
        other = None
    return result, other


def save_chain_step(result, cov, i, counter_sample, counter_dist, other):

    with open(os.path.join(g.path['output'], 'result.dat'), 'a+') as result_file:
        result_file.write(f'{result}\n')
    with open(os.path.join(g.path['output'], 'covariance'), 'a+') as cov_file:
        cov_file.write(f'{cov.flatten()}\n')
    with open(os.path.join(g.path['output'], 'counter'), 'a+') as counter_file:
        counter_file.write(f'{i} {counter_sample} {counter_dist}\n')

    cp, u, uv, u_surface, x_shock = other
    with open(os.path.join(g.path['output'], 'cp_all.bin'), 'ab') as f:
        f.write(bytearray(cp))
    with open(os.path.join(g.path['output'], 'u_slice.bin'), 'ab') as f:
        f.write(bytearray(u))
    with open(os.path.join(g.path['output'], 'uv_slice.bin'), 'ab') as f:
        f.write(bytearray(uv))
    with open(os.path.join(g.path['output'], 'u_surface.bin'), 'ab') as f:
        f.write(bytearray(u_surface))
    with open(os.path.join(g.path['output'], 'x_separation.bin'), 'ab') as f:
        f.write(bytearray(x_shock))
    return


def save_failed_step(result, i, counter_sample, counter_dist):

    with open(os.path.join(g.path['output'], 'not_accepted.dat'), 'a+') as result_file:
        result_file.write(f'{result}\n')
    with open(os.path.join(g.path['output'], 'counter'), 'a+') as counter_file:
        counter_file.write(f'{i} {counter_sample} {counter_dist}\n')

    return


def restart_chain(result_file, N_params, t0, c_init):
    logging.info(f"restart: {os.path.exists(result_file)}")
    done = 0
    if os.path.exists(result_file):
        with open(result_file) as f:
            lines = f.readlines()
        done = len(lines) - 1
        g.c_array = np.empty((done + 1, N_params))
        for i, line in enumerate(lines):
            c = np.fromstring(line[1:-1], dtype=float, sep=',')[:N_params]
            g.c_array[i] = c
        last_result = np.fromstring(lines[-1][1:-1], dtype=float, sep=',')
        last_c = last_result[:N_params]
        logging.info(f'Restart from step {done} with c = {last_c}')
        c_init = last_c
        if done >= t0:
            with open(os.path.join(g.path['output'], 'covariance')) as f:
                lines = f.readlines()
            cov = []
            for line in lines[-N_params:]:
                 cov.append(np.fromstring(line[1:-1], dtype=float, sep=' '))
            g.std = np.array(cov)

    g.restart_chain = done
    np.savetxt(os.path.join(g.path['output'], f'C_start_{done}'), c_init)
