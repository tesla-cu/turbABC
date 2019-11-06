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
    overflow.run_overflow(i)    # run output
    cp, u, uv = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.indices)
    sum_stat_cp = calc_sum_stat(g.Grid.grid_x, cp, g.Truth.cp[:, 0])
    sum_stat_u = np.empty((0,))
    sum_stat_uv = np.empty((0,))
    for x in range(len(u)):
        stat_u = calc_sum_stat(g.Grid.grid_y[x], u[x], g.Truth.u[x][:, 0])
        sum_stat_u = np.hstack((sum_stat_u, stat_u))
        stat_uv = calc_sum_stat(g.Grid.grid_y[x], uv[x], g.Truth.uv[x][:, 0])
        sum_stat_uv = np.hstack((sum_stat_uv, stat_uv))
    sum_stat = np.hstack((sum_stat_cp, sum_stat_u, sum_stat_uv))
    print('!!!!!!!!!!!!', len(sum_stat))
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    result = np.hstack((c, sum_stat, err)).tolist()
    return result
