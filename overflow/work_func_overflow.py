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
