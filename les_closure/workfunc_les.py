import numpy as np

import pyabc.glob_var as g
import pyabc.distance as dist


if g.norm_order == 1:
    calc_err = dist.calc_err_norm1
elif g.norm_order == 2:
    calc_err = dist.calc_err_norm2


####################################################################################################################
# sum_stat_from_C
####################################################################################################################
def abc_work_function(c):

    g.LesModel.calc_modeled_sigma(c)
    sum_stat = g.SumStat.calc_sum_stat(g.LesModel.sigma, g.LesModel.S_random)
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    return np.hstack((c, sum_stat, err)).tolist()



