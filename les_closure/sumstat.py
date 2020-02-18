import numpy as np
import os
import logging
import pyabc.utils as utils
import pyabc.glob_var as g


class TruthData(object):
    def __init__(self, valid_folder, data_params, sumstat_params):

        # self.delta = delta
        dns_data = self.load_HIT_data(valid_folder, data_params)
        dx = np.array([data_params['lx'] / data_params['N_point']] * 3)
        self.S = self.calc_strain_tensor(dns_data, dx)
        self.R = self.calc_rotation_tensor(dns_data, dx)
        self.sumstat_true = dict()
        if 'sigma_pdf_log' in sumstat_params['sumstat']:
            self.sumstat_true = self.deviatoric_stresses_pdf(dns_data, sumstat_params)
            production = self.production_rate_pdf(dns_data, sumstat_params)
            with open(os.path.join(g.path['output'], 'sum_stat_true'), 'wb') as f:
                np.savetxt(f, [self.sumstat_true['uu'], self.sumstat_true['uv'], self.sumstat_true['uw'], production])

        if 'production_pdf_log' in sumstat_params['sumstat']:
            sigma = self.deviatoric_stresses_pdf(dns_data, sumstat_params)
            self.sumstat_true['prod'] = self.production_rate_pdf(dns_data, sumstat_params)
            with open(os.path.join(g.path['output'], 'sum_stat_true'), 'wb') as f:
                np.savetxt(f, [sigma['uu'], sigma['uv'], sigma['uw'], self.sumstat_true['prod']])

        if 'production_mean' in sumstat_params['sumstat']:
            self.sum_stat_true = self.production_rate_mean(dns_data)


    @staticmethod
    def field_gradient(field, dx):
        """Calculate tensor of gradients of self.field.
        :return:      dictionary of gradient tensor
        """
        grad = dict()
        grad['uu'], grad['uv'], grad['uw'] = np.gradient(field['u'], dx[0], dx[1], dx[2])
        grad['vu'], grad['vv'], grad['vw'] = np.gradient(field['v'], dx[0], dx[1], dx[2])
        grad['wu'], grad['wv'], grad['ww'] = np.gradient(field['w'], dx[0], dx[1], dx[2])
        return grad

    def calc_strain_tensor(self, field, dx):
        """Calculate strain tensor S_ij = 1/2(du_i/dx_j+du_j/dx_i) of given field.
        :return:      dictionary of strain tensor
        """
        A = self.field_gradient(field, dx)
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
        return tensor

    def calc_rotation_tensor(self, field, dx):
        """Calculate rotation tensor R_ij = 1/2(du_i/dx_j-du_j/dx_i) of given field.
        :return:       dictionary of rotation tensor
        """
        A = self.field_gradient(field, dx)
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] - A[j + i])
        return tensor

    def calc_tau(self, field):
        """Calculate Reynolds stresses field using DNS data.
            tau_ij = \tilde{u_iu_j} - \tilde{u_i}\tilde{u_j}
        :return:     dictionary of Reynolds stresses tensor
        """
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i + j] = field[i + j] - np.multiply(field[i], field[j])
        return tau

    def deviatoric_stresses_pdf(self, field, pdf_params):
        """Calculate pdf of deviatoric stresses using DNS data.
            sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
        :return:     dictionary of log pdfs of deviatoric stresses
        """
        tau = self.calc_tau(field)
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        sigma = tau
        log_sigma_pdf = dict()
        for key, value in sigma.items():
            _, sigma_pdf = utils.pdf_from_array_with_x(value, pdf_params['bins'], pdf_params['domain'])
            log_sigma_pdf[key] = utils.take_safe_log(sigma_pdf)

        return log_sigma_pdf

    def production_rate_pdf(self, field, pdf_params):
        """Calculate kinetic energy production rate using DNS data.
            P = -\tau_ij \partial\tilde{u_i}/\partial x_j
        :return: log of pdf of production rate
        """
        tau = self.calc_tau(field)
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        prod_rate = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                prod_rate += tau[i + j]*self.S[i + j]
        _, prod_rate_pdf = utils.pdf_from_array_with_x(prod_rate, pdf_params['bins'], pdf_params['domain_production'])
        log_prod_pdf = utils.take_safe_log(prod_rate_pdf)
        return log_prod_pdf

    def production_rate_mean(self, field):
        """Calculate kinetic energy production rate using DNS data.
            P = \tau_ij \tilde{S_ij}
        :return: mean of production rate (single value)
        """
        tau = self.calc_tau()
        prod_rate = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                prod_rate += tau[i + j]*self.S[i + j]
        prod_rate_mean = np.mean(prod_rate)
        return prod_rate_mean


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


def load_HIT_data(valid_folder, case_params):
    datafile = dict()
    # 'JHU_data':
    datafile['u'] = os.path.join(valid_folder, 'HIT_u.bin')
    datafile['v'] = os.path.join(valid_folder, 'HIT_v.bin')
    datafile['w'] = os.path.join(valid_folder, 'HIT_w.bin')
    type_of_bin_data = np.float32
    # 'CU_data'
    # datafile['u'] = os.path.join(data_params['data_path'], 'Velocity1_003.rst')
    # datafile['v'] = os.path.join(data_params['data_path'], 'Velocity2_003.rst')
    # datafile['w'] = os.path.join(data_params['data_path'], 'Velocity3_003.rst')
    # type_of_bin_data = np.float64
    logging.info('Load HIT data')
    HIT_data = dict()
    data_shape = (case_params['N_point'], case_params['N_point'], case_params['N_point'])
    for i in ['u', 'v', 'w']:
        HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
    for key, value in HIT_data.items():
        HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])
    return HIT_data



def calc_sum_stat(x, y, valid_data_x):
    return np.interp(valid_data_x, x, y)