import numpy as np
from numpy.linalg import norm as norm2


def calc_err_inf(sum_stat, sumstat_true):
    return np.max(np.abs(sum_stat - sumstat_true))


def calc_err_norm1(sum_stat, sumstat_true):
    return np.sum(np.abs(sum_stat - sumstat_true))


def calc_err_norm2(sum_stat, sumstat_true):
    return norm2(sum_stat - sumstat_true)