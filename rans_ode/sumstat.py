import numpy as np
import os


class TruthData(object):
    def __init__(self, valid_folder, case):

        if 'impulsive' in case:
            self.axi_exp_k = np.loadtxt(os.path.join(valid_folder, 'axi_exp_k.txt'))
            self.axi_exp_a = np.loadtxt(os.path.join(valid_folder, 'axi_exp_b.txt'))
            self.axi_con_k = np.loadtxt(os.path.join(valid_folder, 'axi_con_k.txt'))
            self.axi_con_a = np.loadtxt(os.path.join(valid_folder, 'axi_con_b.txt'))
            self.shear_k = np.loadtxt(os.path.join(valid_folder, 'shear_k.txt'))
            self.plane_k = np.loadtxt(os.path.join(valid_folder, 'plane_k.txt'))
            self.plane_a = np.loadtxt(os.path.join(valid_folder, 'plane_b.txt'))
            self.sumstat_true = np.hstack((self.axi_exp_k[:, 1], self.axi_con_k[:, 1],
                                           self.shear_k[:, 1], self.plane_k[:, 1]))
            # self.sumstat_true_a = np.hstack((2*self.axi_exp_b[:, 1], 2*self.axi_con_b[:, 1], 2*self.plane_b[:, 1]))
        if 'periodic' in case:
            self.periodic_k = []
            for i in range(5):
                self.periodic_k.append(np.loadtxt(os.path.join(valid_folder, 'period{}_k.txt'.format(i+1))))
            self.sumstat_true = np.array([item[1] for case in self.periodic_k for item in case])
        if 'decay' in case:
            self.sumstat_true = 2 * np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[:, 1]
            self.decay_a11 = np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[:12]
            self.decay_a22 = np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[12:24]
            self.decay_a33 = np.loadtxt(os.path.join(valid_folder, 'decay_exp_a.txt'))[24:]

        if 'strain-relax' in case:
            self.strain_relax_a11 = np.loadtxt(os.path.join(valid_folder, 'strain_relax_b11.txt'))
            self.sumstat_true = self.strain_relax_a11[:, 1]


def calc_sum_stat(x, y, valid_data_x):
    points = np.interp(valid_data_x, x, y)
    points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))
    return points


def calc_err_norm1(x, y, valid_data_x, valid_data_y):
    points = np.interp(valid_data_x, x, y)
    points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))   # adding gaussian noise
    diff = points - valid_data_y
    return np.max(np.abs(diff))


def calc_err_norm2(x, y, valid_data_x, valid_data_y):
    points = np.interp(valid_data_x, x, y)
    points += np.random.normal(loc=0.0, scale=0.0008, size=len(points))   # adding gaussian noise
    diff = norm2(points - valid_data_y)
    return diff