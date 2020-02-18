import logging
import numpy as np
import os
from scipy.integrate import odeint
from time import time

import rans_ode.ode as rans
from rans_ode.sumstat import calc_sum_stat
import pyabc.glob_var as g
import pyabc.distance as dist
from pyabc.utils import take_safe_log10

from sumstat import load_HIT_data

if g.norm_order == 1:
    calc_err = dist.calc_err_norm1
elif g.norm_order == 2:
    calc_err = dist.calc_err_norm2


def define_work_function():
    if g.case == 'impulsive':
        work_function = abc_work_function_impulsive
    elif g.case == 'periodic':
        work_function = abc_work_function_periodic
    elif g.case == 'decay':
        work_function = abc_work_function_decay
    elif g.case == 'strain-relax':
        work_function = abc_work_function_strain_relax
    else:
        logging.error('Unknown work function {}'.format(g.case))
        exit()
    logging.info('Work function is {}'.format(g.case))
    return work_function


########################################################################################################################
#
########################################################################################################################
def abc_work_function_impulsive(c):

    u0 = [1, 1, 0, 0, 0, 0, 0, 0]
    # axisymmetric expansion
    tspan = np.linspace(0, 1.6 / np.abs(g.Strain.axi_exp[0]), 200)

    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.axi_exp), atol=1e-8, mxstep=200)
    sum_stat_k1 = calc_sum_stat(np.abs(g.Strain.axi_exp[0]) * tspan, Ynke[:, 0], g.Truth.axi_exp_k[:, 0])
    # sum_stat_b1 = calc_sum_stat(np.abs(g.Strain_axi_exp[0]) * tspan, Ynke[:, 2], g.Truth.axi_exp_b[:, 0])

    # axisymmetric contraction
    tspan = np.linspace(0, 1.6 / np.abs(g.Strain.axi_con[0]), 200)
    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.axi_con), atol=1e-8, mxstep=200)
    sum_stat_k2 = calc_sum_stat(np.abs(g.Strain.axi_con[0]) * tspan, Ynke[:, 0], g.Truth.axi_con_k[:, 0])
    # sum_stat_b2 = calc_sum_stat(np.abs(g.Strain_axi_con[0]) * tspan, Ynke[:, 2], g.Truth.axi_con_b[:, 0])

    # pure shear
    tspan = np.linspace(0, 5.2 / (2*g.Strain.pure_shear[3]), 200)
    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.pure_shear), atol=1e-8, mxstep=200)
    sum_stat_k3 = calc_sum_stat(2 * g.Strain.pure_shear[3] * tspan, Ynke[:, 0], g.Truth.shear_k[:, 0])

    # plane strain
    tspan = np.linspace(0, 1.6 / g.Strain.plane_strain[0], 200)
    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.plane_strain), atol=1e-8, mxstep=200)
    sum_stat_k4 = calc_sum_stat(np.abs(g.Strain.plane_strain[0]) * tspan, Ynke[:, 0], g.Truth.plane_k[:, 0])
    # sum_stat_b4 = calc_sum_stat(np.abs(g.Strain_plane_strain[0]) * tspan, Ynke[:, 2], g.Truth.plane_b[:, 0],)

    sum_stat = np.hstack((sum_stat_k1, sum_stat_k2, sum_stat_k3, sum_stat_k4))
    noise = np.random.normal(loc=0.0, scale=0.0008, size=len(sum_stat))
    err = calc_err(sum_stat+noise, g.Truth.sumstat_true)
    result = np.hstack((c, sum_stat, err)).tolist()
    return result


def abc_work_function_periodic(c):
    time1 = time()
    s0 = 3.3
    beta = [0.125, 0.25, 0.5, 0.75, 1]
    u0 = [1, 1, 0, 0, 0, 0, 0, 0]
    # Periodic shear(five different frequencies)
    tspan = np.linspace(0, 50/s0, 500)
    sum_stat = []
    for i in range(5):
        Ynke = odeint(rans.rans_periodic, u0, tspan, args=(c, s0, beta[i], StrainTensor.periodic_strain),
                      atol=1e-8, mxstep=200)
        ss_new = calc_sum_stat(s0 * tspan, take_safe_log10(Ynke[:, 0]), g.Truth.periodic_k[i][:, 0])
        sum_stat = np.hstack((sum_stat, ss_new))
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    result = np.hstack((c, err)).tolist()
    time2 = time()
    print(time2-time1, 'nan/inf = {}'.format(np.isnan(err) or np.isinf(err)), err)
    return result


def abc_work_function_decay(c):

    u0 = [1, 1, 0.36, -0.08, -0.28, 0, 0, 0]
    tspan = np.linspace(0, 0.3, 200)
    Ynke = odeint(rans.rans_decay, u0, tspan, args=(c, ), atol=1e-8, mxstep=200)
    sum_stat1 = calc_sum_stat(tspan, Ynke[:, 2], g.Truth.decay_a11[:, 0])
    sum_stat2 = calc_sum_stat(tspan, Ynke[:, 3], g.Truth.decay_a22[:, 0])
    sum_stat3 = calc_sum_stat(tspan, Ynke[:, 4], g.Truth.decay_a33[:, 0])
    sum_stat = np.hstack((sum_stat1, sum_stat2, sum_stat3))
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    result = np.hstack((c, sum_stat, err)).tolist()
    return result


def abc_work_function_strain_relax(c):

    u0 = [1, 1, 0.36, -0.08, -0.28, 0, 0, 0]
    # strain-relaxation
    tspan = np.linspace(0.0775, 0.953, 500)
    Ynke = odeint(rans.rans_strain_relax, u0, tspan, args=(c, g.Strain.strain_relax), atol=1e-8, mxstep=200)
    sum_stat = calc_sum_stat(tspan, Ynke[:, 2], g.Truth.strain_relax_a11[:, 0])
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    result = np.hstack((c, sum_stat, err)).tolist()
    return result


########################################################################################################################
#
########################################################################################################################
class DataFiltered(object):
    def __init__(self, valid_folder, case_params):


        dx = np.array([case_params['lx'] / case_params['N_point']] * 3)
        DNS_delta = case_params['lx'] / case_params['N_point']
        LES_delta = 1 / case_params['LES_scale'

        HIT_data = load_HIT_data(valid_folder, case_params)
        logging.info('Filter HIT data')
        LES_data = utils.filter3d(data=HIT_data, scale_k=case_params['LES_scale'],
                                  dx=dx, N_points=[case_params['N_point']] * 3)
        logging.info('Writing file')
        np.savez(os.path.join(data_params['data_path'], 'LES.npz'), **LES_data)
        logging.info('Create LES class')
        g.LES = data.Data(HIT_data, DNS_delta, dx, pdf_params)
        # np.savez(os.path.join(data_params['data_path'], 'sum_stat_true.npz'), **g.sum_stat_true)
        del HIT_data
        LES_delta = 1 / case_params['LES_scale']
        logging.info('Create TEST class')
        g.TEST = data.Data(LES_data, LES_delta, dx, pdf_params)
        del LES_delta


class DataSparse(object):

    def __init__(self, path, load, data=None, n_training=None):

        if load:
            sparse_data = np.load(os.path.join(path, 'TEST_sp.npz'))
            self.delta = sparse_data['delta'].item()
            self.S = sparse_data['S'].item()
            self.R = sparse_data['R'].item()
            g.sum_stat_true = np.load(os.path.join(path, 'sum_stat_true.npz'))
            logging.info('Training data shape is ' + str(self.S['uu'].shape))
        else:
            M = n_training
            self.delta = data.delta
            logging.info('Sparse data')
            self.S = self.sparse_dict(data.S, M)
            self.R = self.sparse_dict(data.R, M)
            logging.info('Training data shape is ' + str(self.S['uu'].shape))
            np.savez(os.path.join(path, 'TEST_sp.npz'), delta=self.delta, S=self.S, R=self.R)
            # np.savez(os.path.join(path, 'sum_stat_true.npz'), **g.sum_stat_true)

    def sparse_array(self, data_value, M):

        if data_value.shape[0] % M:
            logging.warning('Error: DataSparse.sparse_dict(): Nonzero remainder')
        n_th = int(data_value.shape[0] / M)
        sparse_data = data_value[::n_th, ::n_th, ::n_th].copy()
        return sparse_data

    def sparse_dict(self, data_dict, M):

        sparse = dict()
        for key, value in data_dict.items():
            sparse[key] = self.sparse_array(value, M)
        return sparse