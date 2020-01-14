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
    cp, sum_stat_u, sum_stat_uv, u_slice, uv_slice = overflow.read_data_from_overflow(g.job_folder, g.Grid.grid, g.Grid.x_slices, g.Grid.y_slices)
    sum_stat_cp = calc_sum_stat(g.Grid.grid_x[::-1], cp[::-1], g.Truth.cp[:, 0])
    sum_stat = np.hstack((sum_stat_cp, sum_stat_u, sum_stat_uv))
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    result = np.hstack((c, sum_stat, err)).tolist()
    return result, cp, u_slice, uv_slice
