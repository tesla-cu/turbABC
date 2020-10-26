import numpy as np
import os
from scipy.integrate import odeint
import rans_ode.ode as rans


class TruthData(object):
    def __init__(self, valid_folder, case):

        self.sumstat_true = np.empty((0))
        self.norm = np.empty((0))
        if 'impulsive_k' in case:
            self.axi_exp_k = np.loadtxt(os.path.join(valid_folder, 'axi_exp_k.txt'))
            self.axi_con_k = np.loadtxt(os.path.join(valid_folder, 'axi_con_k.txt'))
            self.shear_k = np.loadtxt(os.path.join(valid_folder, 'shear_k.txt'))
            self.plane_k = np.loadtxt(os.path.join(valid_folder, 'plane_k.txt'))
            norm_k = np.hstack((self.normalize(self.axi_exp_k[:, 1]), self.normalize(self.axi_con_k[:, 1]),
                                self.normalize(self.shear_k[:, 1]), self.normalize(self.plane_k[:, 1])))
            sumstat_impulsive = np.hstack((self.axi_exp_k[:, 1], self.axi_con_k[:, 1],
                                           self.shear_k[:, 1], self.plane_k[:, 1]))/norm_k
            self.sumstat_true = np.hstack((self.sumstat_true, sumstat_impulsive))
            self.norm = np.hstack((self.norm, norm_k))

        if 'impulsive_a' in case:
            self.axi_exp_a = np.loadtxt(os.path.join(valid_folder, 'axi_exp_b.txt'))
            self.axi_con_a = np.loadtxt(os.path.join(valid_folder, 'axi_con_b.txt'))
            self.plane_a11 = np.loadtxt(os.path.join(valid_folder, 'plane_b.txt'))
            self.plane_a22 = np.loadtxt(os.path.join(valid_folder, 'plane_b22.txt'))
            self.axi_exp_a[:, 1] *= 2
            self.axi_con_a[:, 1] *= 2
            self.plane_a11[:, 1] *= 2
            self.plane_a22[:, 1] *= 2
            norm_a = np.hstack((self.normalize(self.axi_exp_a[:, 1]), self.normalize(self.axi_con_a[:, 1]),
                                self.normalize(self.plane_a11[:, 1]), self.normalize(self.plane_a22[:, 1])))
            sumstat_true_a = np.hstack((self.axi_exp_a[:, 1], self.axi_con_a[:, 1],
                                        self.plane_a11[:, 1], self.plane_a22[:, 1]))/norm_a
            self.sumstat_true = np.hstack((self.sumstat_true, sumstat_true_a))
            self.norm = np.hstack((self.norm, norm_a))

        if 'periodic' in case:
            self.periodic_k = []
            norm_periodic = np.empty(0)
            self.norm = norm_periodic
            for i in range(5):
                data = np.loadtxt(os.path.join(valid_folder, 'period{}_k.txt'.format(i + 1)))
                self.periodic_k.append(data)
                norm = np.max(np.log10(data[:, 1])) * np.ones_like(data[:, 1]) * len(data)
                norm_periodic = np.hstack((norm_periodic, norm))
            sumstat_periodic = np.array([np.log10(item[1]) for case in self.periodic_k for item in case])
            sumstat_periodic = sumstat_periodic/norm_periodic
            self.sumstat_true = np.hstack((self.sumstat_true, sumstat_periodic))
            self.norm = np.hstack((self.norm, norm_periodic))

        if 'validation_exp' in case:
            self.validation_exp = np.loadtxt(os.path.join(valid_folder, 'period3_k.txt'))
            self.sumstat_true = np.hstack((self.sumstat_true,  np.log10(self.validation_exp[:, 1])))

        if 'validation_nominal' in case:
            def periodic_strain(t, s0, beta):
                S = np.zeros(6)
                S[3] = (s0 / 2) * np.sin(beta * s0 * t)
                return S
            s0 = 3.3
            beta = 0.5    # frequancy w/Smax
            u0 = [1, 1, 0, 0, 0, 0, 0, 0]
            tspan = np.linspace(0, 50 / s0, 500)
            nominal_values = np.array([1.5, 0.8, 1.44, 1.83])
            t_exp = np.loadtxt(os.path.join(valid_folder, 'period5_k.txt'))[:, 0]
            Ynke = odeint(rans.rans_periodic, u0, tspan,
                          args=(nominal_values, s0, beta, periodic_strain), atol=1e-8, mxstep=200)
            sumstat = calc_sum_stat(s0 * tspan, Ynke[:, 0], t_exp)
            self.validation_nominal = np.empty((len(t_exp), 2))
            self.validation_nominal[:, 0] = t_exp
            self.validation_nominal[:, 1] = sumstat
            self.sumstat_true = np.hstack((self.sumstat_true,  np.log10(sumstat)))

        if 'decay' in case:

            self.decay_a11 = np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[:12]
            self.decay_a22 = np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[12:24]
            self.decay_a33 = np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[24:]
            self.decay_a11[:, 1] *= 2
            self.decay_a22[:, 1] *= 2
            self.decay_a33[:, 1] *= 2
            norm_a = np.hstack((self.normalize(self.decay_a11[:, 1]), self.normalize(self.decay_a22[:, 1]),
                                self.normalize(self.decay_a33[:, 1])))
            sumstat_decay = 2 * np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[:, 1]/norm_a
            self.sumstat_true = np.hstack((self.sumstat_true, sumstat_decay))
            self.norm = np.hstack((self.norm, norm_a))

        if 'strain-relax' in case:
            self.strain_relax_a11 = np.loadtxt(os.path.join(valid_folder, 'strain_relax_b11.txt'))
            sumstat_true = self.strain_relax_a11[:, 1]
            norm = self.normalize(sumstat_true)
            self.sumstat_true = np.hstack((self.sumstat_true, sumstat_true))
            self.norm = np.hstack((self.norm, norm))


    @staticmethod
    def normalize(array):
        return np.max(np.abs(array)) * np.ones_like(array) * np.sqrt(len(array))


def calc_sum_stat(x, y, valid_data_x):
    return np.interp(valid_data_x, x, y)



# def calc_err_norm1(x, y, valid_data_x, valid_data_y):
#     points = np.interp(valid_data_x, x, y)
#     points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))   # adding gaussian noise
#     diff = points - valid_data_y
#     return np.max(np.abs(diff))
#
#
# def calc_err_norm2(x, y, valid_data_x, valid_data_y):
#     points = np.interp(valid_data_x, x, y)
#     points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))   # adding gaussian noise
#     diff = norm2(points - valid_data_y)
#     return diff