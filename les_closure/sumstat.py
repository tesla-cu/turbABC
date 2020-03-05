import numpy as np
import os
# import logging
import pyabc.utils as utils
import pyabc.glob_var as g
import data


class TruthData(object):
    def __init__(self, valid_folder, data_params, sumstat_params):

        dns_data = data.load_HIT_data(valid_folder, data_params['N_point'])
        n_points = [data_params['N_point']] * 3
        dx = np.array([data_params['lx'] / data_params['N_point']] * 3)
        les_data = data.DataFiltered.filter3d(data=dns_data, scale_k=data_params['LES_scale'], dx=dx, N_points=n_points)
        sigma_field = self.calc_sigma_field(les_data)
        strain_filtered = data.calc_strain_tensor(les_data, dx)
        Sumstat = SummaryStatistics(sumstat_params)
        sigma = Sumstat.deviatoric_stresses_pdf(sigma_field)
        production = Sumstat.production_rate_pdf(sigma_field, strain_filtered)
        with open(os.path.join(g.path['output'], 'sum_stat_true'), 'wb') as f:
            np.savetxt(f, np.hstack((sigma, production)))

        self.sumstat_true = np.empty((0,))
        if 'sigma_pdf_log' in sumstat_params['sumstat']:
            for ij in ['uu', 'vv', 'ww', 'uv', 'uw', 'vw']:
                self.sumstat_true = np.hstack((self.sumstat_true, sigma[ij]))
        if 'production_pdf_log' in sumstat_params['sumstat']:
            self.sumstat_true = np.hstack((self.sumstat_true, production))
        if 'production_mean' in sumstat_params['sumstat']:
            production_mean = Sumstat.production_rate_mean(sigma_field, strain_filtered)
            self.sumstat_true = np.hstack((self.sumstat_true, production_mean))

    @staticmethod
    def calc_sigma_field(field):
        """Calculate Reynolds stresses field using DNS data.
            tau_ij = \tilde{u_iu_j} - \tilde{u_i}\tilde{u_j}
        :return:     dictionary of Reynolds stresses tensor
        """
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i + j] = field[i + j] - np.multiply(field[i], field[j])
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        return tau


class SummaryStatistics(object):
    def __init__(self, sumstat_params):
        self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        self.bins = sumstat_params['bins']
        self.domain = sumstat_params['domain']
        self.domain_production = sumstat_params['domain_production']
        self.sumstat_type = sumstat_params['sumstat']

        if self.sumstat_type == 'sigma_pdf_log':
            self.calc_sum_stat = self.deviatoric_stresses_pdf
        elif self.sumstat_type == 'production_pdf_log':
            self.calc_sum_stat = self.production_rate_pdf
        elif self.sumstat_type == 'production_mean':
            self.calc_sum_stat = self.production_rate_mean

    def deviatoric_stresses_pdf(self, sigma_field, S=None):
        """Calculate pdf of deviatoric stresses using DNS data.
            sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
        :return:     dictionary of log pdfs of deviatoric stresses
        """
        log_sigma_pdf = np.empty(len(self.elements_in_tensor)*self.bins)
        for i, key in enumerate(self.elements_in_tensor):
            _, sigma_pdf = utils.pdf_from_array_with_x(sigma_field[key], self.bins, self.domain)
            log_sigma_pdf[i*self.bins : (i+1)*self.bins] = utils.take_safe_log(sigma_pdf)
        return log_sigma_pdf

    def production_rate_pdf(self, sigma_field, S_filtered):
        """Calculate kinetic energy production rate using DNS data.
            P = -\tau_ij \partial\tilde{u_i}/\partial x_j
        :return: log of pdf of production rate
        """
        prod_rate = 0
        for ij in ['uu', 'vv', 'ww', 'uv', 'uw', 'vw', 'uv', 'uw', 'vw']:
            # double count off-diagonal terms (homogeneous data)
            prod_rate += sigma_field[ij]*S_filtered[ij]
        _, prod_rate_pdf = utils.pdf_from_array_with_x(prod_rate, self.bins, self.domain_production)
        log_prod_pdf = utils.take_safe_log(prod_rate_pdf)
        return log_prod_pdf

    def production_rate_mean(self, sigma_field, S_filtered):
        """Calculate kinetic energy production rate using DNS data.
            P = \tau_ij \tilde{S_ij}
        :return: mean of production rate (single value)
        """
        prod_rate = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                prod_rate += sigma_field[i + j]*S_filtered[i + j]
        prod_rate_mean = np.mean(prod_rate)
        return prod_rate_mean
    #
    # def both_pdf(self, C):
    #     """ Create array of 7 pdfs (6 sigma pdf and 1 production pdf). """
    #     self.sigma_field_from_C(C, 1)
    #     both_pdf = np.empty((len(self.elements_in_tensor)+1, self.pdf_params['bins']))
    #     for ind, key in enumerate(self.elements_in_tensor):
    #         both_pdf[ind] = utils.pdf_from_array(self.sigma[key], self.pdf_params['bins'], self.pdf_params['domain'])
    #     production = 0
    #     for key, value in self.sigma.items():
    #         production += self.sigma[key] * self.S[key]
    #     both_pdf[-1] = np.array(
    #         [utils.pdf_from_array(production, self.pdf_params['bins'], self.pdf_params['domain_production'])])
    #     return both_pdf#

#
# def calc_sum_stat(sigma_field, S_filtered):
#     return g.sumstat.sumstat_func(sigma_field, S_filtered)