import numpy as np
import os
from scipy.integrate import odeint
from time import time

import rans_ode as rans
import pyabc.glob_var as g
import pyabc.distance as dist
from pyabc.utils import take_safe_log10
from sumstat import calc_sum_stat

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
    elif g.case == 'strain_relax':
        work_function = abc_work_function_strain_relax
    else:
        logging.error('Unknown work function {}'.format(g.case))
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

    # err[0] = calc_err(np.abs(g.Strain.axi_exp[0]) * tspan, Ynke[:, 0], g.Truth.axi_exp_k[:, 0], g.Truth.axi_exp_k[:, 1])
    # err[9] = calc_err(np.abs(S_axi_exp[0]) * Tnke, Ynke[:, 2], axi_exp_b[:, 0], 2 * axi_exp_b[:, 1])

    # axisymmetric contraction
    tspan = np.linspace(0, 1.6 / np.abs(g.Strain.axi_con[0]), 200)
    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.axi_con), atol=1e-8, mxstep=200)
    sum_stat_k2 = calc_sum_stat(np.abs(g.Strain.axi_con[0]) * tspan, Ynke[:, 0], g.Truth.axi_con_k[:, 0])
    # sum_stat_b2 = calc_sum_stat(np.abs(g.Strain_axi_con[0]) * tspan, Ynke[:, 2], g.Truth.axi_con_b[:, 0])

    # err[1] = calc_err(np.abs(g.Strain.axi_con[0]) * tspan, Ynke[:, 0], g.Truth.axi_con_k[:, 0], g.Truth.axi_con_k[:, 1])
    # # err[10] = calc_err(np.abs(S_axi_con[0]) * Tnke, Ynke[:, 2], axi_con_b[:, 0], 2 * axi_con_b[:, 1])

    # pure shear
    tspan = np.linspace(0, 5.2 / (2*g.Strain.pure_shear[3]), 200)
    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.pure_shear), atol=1e-8, mxstep=200)
    sum_stat_k3 = calc_sum_stat(2 * g.Strain.pure_shear[3] * tspan, Ynke[:, 0], g.Truth.shear_k[:, 0])
    # err[2] = calc_err(2 * g.Strain.pure_shear[3] * tspan, Ynke[:, 0], g.Truth.shear_k[:, 0], g.Truth.shear_k[:, 1])

    # plane strain
    tspan = np.linspace(0, 1.6 / g.Strain.plane_strain[0], 200)
    Ynke = odeint(rans.rans_impulsive, u0, tspan, args=(c, g.Strain.plane_strain), atol=1e-8, mxstep=200)
    sum_stat_k4 = calc_sum_stat(np.abs(g.Strain.plane_strain[0]) * tspan, Ynke[:, 0], g.Truth.plane_k[:, 0])
    # sum_stat_b4 = calc_sum_stat(np.abs(g.Strain_plane_strain[0]) * tspan, Ynke[:, 2], g.Truth.plane_b[:, 0],)

    # err[3] = calc_err(np.abs(g.Strain.plane_strain[0]) * tspan, Ynke[:, 0], g.Truth.plane_k[:, 0], g.Truth.plane_k[:, 1])
    # # err[7] = calc_err(np.abs(S_plane_strain[0]) * Tnke, Ynke[:, 2], plane_b[:, 0], 2 * plane_b[:, 1])

    sum_stat = np.hstack((sum_stat_k1, sum_stat_k2, sum_stat_k3, sum_stat_k4))
    err = calc_err(sum_stat, g.Truth.sumstat_true)
    result = np.hstack((c, sum_stat, err)).tolist()
    return result


def abc_work_function_periodic(c):
    time1 = time()
    s0 = 3.3
    beta = [0.125, 0.25, 0.5, 0.75, 1]
    err = np.zeros(5)
    u0 = [1, 1, 0, 0, 0, 0, 0, 0]
    # Periodic shear(five different frequencies)
    tspan = np.linspace(0, 50/s0, 500)
    for i in range(5):
        Ynke = odeint(rans.rans_periodic, u0, tspan, args=(c, s0, beta[i]), atol=1e-8, mxstep=200)
        err[i] = calc_err(s0 * tspan, take_safe_log10(Ynke[:, 0]), g.Truth.periodic_k[i][:, 0], g.Truth.periodic_k[i][:, 1])
    result = np.hstack((c, err)).tolist()
    time2 = time()
    print(time2-time1)
    print(time2-time1, 'nan/inf = {}'.format((True in np.isnan(err)) or (True in np.isinf(err))))
    return result


def abc_work_function_decay(c):

    err = np.zeros(3)
    u0 = [1, 1, 0.36, -0.08, -0.28, 0, 0, 0]
    # decay
    tspan = np.linspace(0, 45, 500)
    Ynke  = odeint(rans.rans_decay, u0, tspan, args=(c,), atol=1e-8, mxstep=200)
    err[0] = calc_err(tspan, Ynke[:, 2], g.Truth.decay_a11[:, 0], 2 * g.Truth.decay_a11[:, 1])
    err[1] = calc_err(tspan, Ynke[:, 3], g.Truth.decay_a22[:, 0], 2 * g.Truth.decay_a22[:, 1])
    err[2] = calc_err(tspan, Ynke[:, 4], g.Truth.decay_a33[:, 0], 2 * g.Truth.decay_a33[:, 1])
    result = np.hstack((c, np.sum(err))).tolist()
    return result


def abc_work_function_strain_relax(c):

    u0 = [1, 1, 0.36, -0.08, -0.28, 0, 0, 0]
    # strain-relaxation
    tspan = np.linspace(0, 0.95, 500)
    Tnke, Ynke = odeint(rans.rans_strain_relax, u0, tspan, args=c, atol=1e-8, mxstep=200)
    err = calc_err(Tnke, Ynke[:, 2], g.Truth.strain_relax_a11[:, 0], g.Truth.strain_relax_a11[:, 1])
    result = np.hstack((c, np.sum(err))).tolist()
    return result




class StrainTensor(object):
    def __init__(self, valid_folder):

        self.true_strain = np.loadtxt(os.path.join(valid_folder, 'ske.txt'))
        self.axi_exp = self.create_axisymmetric_expansion()
        self.axi_con = self.create_axisymmetric_contraction()
        self.pure_shear = self.create_pure_shear()
        self.plane_strain = self.create_plane_strain()


    @staticmethod
    def create_axisymmetric_expansion():
        S = np.zeros(6)
        s = -(1 / 2.45)
        S[0] = s  # S11
        S[1] = -s / 2  # S22
        S[2] = -s / 2  # S33
        return S

    @staticmethod
    def create_axisymmetric_contraction():
        S = np.zeros(6)
        s = (1 / 0.179)
        S[0] = s
        S[1] = -s / 2
        S[2] = -s / 2
        return S

    @staticmethod
    def create_pure_shear():
        S = np.zeros(6)
        S[3] = (1 / 0.296) / 2  # S12
        return S

    @staticmethod
    def create_plane_strain():
        S = np.zeros(6)
        S[0] = 1 / 2
        S[1] = -1 / 2
        return S

    @staticmethod
    def periodic_strain(t, s0, beta):
        S = np.zeros(6)
        S[3] = (s0 / 2) * np.sin(beta * s0 * t)
        return S

    def strain_relax(self, t):
        S = np.zeros(6)
        S[0] = np.interp(t, self.true_strain[:, 0], self.true_strain[:, 1])
        S[1] = -S[0]
        return S