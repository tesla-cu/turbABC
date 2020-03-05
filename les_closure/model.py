import logging
import numpy as np


class NonlinearModel(object):
    def __init__(self, data, model_params, n_data_points, calc_strain_flag):

        self.N_params = model_params['N_params']
        self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        self.random = model_params['random']
        self.n_data_points = n_data_points
        self.calc_strain_flag = calc_strain_flag

        self.sigma = dict()
        self.S_les = dict()
        self.S_random = dict()
        for i in self.elements_in_tensor:
            self.sigma[i] = np.zeros(self.random)
            self.S_les[i] = data.S[i].flatten()

        logging.info('Calculate model tensors')
        self.Tensor = dict()
        self.S_mod = self.calc_strain_mod(data)
        for i in range(self.N_params):
            self.Tensor[str(i)] = self.calc_tensor(data, number=i)
            logging.info('Tensor {} , {}'.format(i, self.Tensor[str(i)].keys()))
        del self.S_mod

        # logging.info('Nonlinear model with {}'.format(self.sum_stat_from_C.__name__))

    def calc_modeled_sigma(self, c):
        """Calculate deviatoric part of Reynolds stresses using eddy-viscosity model.
        :param c: list of constant parameters
        :return: dict of modeled stresses tensor (sigma_ij)
        """
        ind = self.rand_ind()
        for i in self.elements_in_tensor:
            self.sigma[i] = np.zeros(len(ind))
            for j in range(self.N_params):
                self.sigma[i] += c[j] * self.Tensor[str(j)][i][ind]
            if self.calc_strain_flag:       # need to calculate production P=\sigma_ij*S_ij
                self.S_random[i] = self.S_les[i][ind]

    def rand_ind(self):
        ind = np.random.randint(0, self.n_data_points ** 3, size=self.random)
        ind = np.unique(ind)
        while len(ind) < 0.99 * self.random:
            ind_add = np.random.randint(0, 256 ** 3, size=(self.random - len(ind)))
            ind = np.unique(np.append(ind, ind_add))
        return ind

    @staticmethod
    def calc_strain_mod(data):
        """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
        :return:       array of |S| in each point of domain
        """
        S_mod_sqr = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                S_mod_sqr += 2 * np.multiply(data.S[i + j], data.S[i + j])
        return np.sqrt(S_mod_sqr)

    def calc_tensor(self, data, number):
        """Calculate tensor T_i for nonlinear viscosity model
        :param data: data class object (sparse data)
        :param number: index of tensor
        :return:  dictionary of tensor
        """

        if number == 0:
            # Calculate tensor Delta^2|S|S_ij for given field
            tensor = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor[i + j] = np.multiply(self.S_mod, data.S[i + j])
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
            return tensor

        elif number == 1:
            # Calculate tensor Delta^2*(S_ikR_kj - R_ikS_kj)  for given field
            tensor = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor[i + j] = 0
                    for k in ['u', 'v', 'w']:
                        tensor[i + j] += np.multiply(data.S[i + k], data.R[k + j]) - \
                                         np.multiply(data.R[i + k], data.S[k + j])
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
            return tensor

        elif number == 2:
            # Calculate tensor Delta^2*(S_ikS_kj - 1/3{S_ikS_ki}delta_ij) for given field
            tensor = dict()
            S_S_inv = 0
            for i in ['u', 'v', 'w']:
                for k in ['u', 'v', 'w']:
                    S_S_inv += np.multiply(data.S[i + k], data.S[k + i])
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor[i + j] = 0
                    for k in ['u', 'v', 'w']:
                        tensor[i + j] += np.multiply(data.S[i + k], data.S[k + j])
                        if i == j:
                            tensor[i + j] -= 1 / 3 * S_S_inv
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
            return tensor

        elif number == 3:
            # Calculate tensor Delta^2(R_ikR_kj - 1/3{R_ikR_ki}delta_ij) for given field
            tensor = dict()
            R_R_inv = 0
            for i in ['u', 'v', 'w']:
                for k in ['u', 'v', 'w']:
                    R_R_inv += np.multiply(data.R[i + k], data.R[k + i])
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor[i + j] = 0
                    for k in ['u', 'v', 'w']:
                        tensor[i + j] += np.multiply(data.R[i + k], data.R[k + j])
                        if i == j:
                            tensor[i + j] -= 1 / 3 * R_R_inv
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
            return tensor

        elif number == 4:
            # Calculate tensor Delta^2/S_mod (R_ikS_klSlj - S_ikS_klRlj) for given field
            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            tensor1[i + j] += data.R[i + k] * data.S[k + l] * data.S[l + j]
                            tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1

        elif number == 5:
            # Calculate tensor Delta^2/S_mod (R_ikR_klSlj + S_ikR_klRlj - 2/3 {S_ikR_klRli}*delta_ij) for given field
            tensor1 = dict()
            S_R_R_inv = 0
            for i in ['u', 'v', 'w']:
                for k in ['u', 'v', 'w']:
                    for l in ['u', 'v', 'w']:
                        S_R_R_inv += data.S[i + k] * data.R[k + l] * data.R[l + i]
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            tensor1[i + j] += data.R[i + k] * data.R[k + l] * data.S[l + j]
                            tensor2 += data.S[i + k] * data.R[k + l] * data.R[l + j]
                    tensor1[i + j] += tensor2
                    if i == j:
                        tensor1[i + j] -= 2 / 3 * S_R_R_inv
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1

        elif number == 6:
            # Calculate tensor Delta^2/S_mod^2 (R_ikS_klR_lm_Rmj - R_ikR_klS_lmR_mj) for given field
            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.R[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + j]
                                tensor2 += data.R[i + k] * data.R[k + l] * data.S[l + m] * data.R[m + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1

        elif number == 7:
            # Calculate tensor Delta^2/S_mod^2 (S_ikR_klS_lm_Smj - S_ikS_klR_lmS_mj)  for given field

            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.S[i + k] * data.R[k + l] * data.S[l + m] * data.S[m + j]
                                tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.S[m + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1

        elif number == 8:
            # Calculate tensor Delta^2/S_mod^2 (R^2S^2 + S^2R^2 - 2/3{S^2R^2}*delta_ij)  for given field
            tensor1 = dict()
            S2_R2_inv = 0
            for i in ['u', 'v', 'w']:
                for k in ['u', 'v', 'w']:
                    for l in ['u', 'v', 'w']:
                        for m in ['u', 'v', 'w']:
                            S2_R2_inv += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + i]
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.R[i + k] * data.R[k + l] * data.S[l + m] * data.S[m + j]
                                tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + j]
                    tensor1[i + j] += tensor2
                    if i == j:
                        tensor1[i + j] -= 2/3*S2_R2_inv
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1

        elif number == 9:
            # Calculate tensor Delta^2/S_mod^3 (RS^2R^2 - R^2S^2R) for given field
            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                for n in ['u', 'v', 'w']:
                                    tensor1[i + j] += data.R[i + k] * data.S[k + l] * \
                                                      data.S[l + m] * data.R[m + n] * data.R[n + j]
                                    tensor2 += data.R[i + k] * data.R[k + l] * \
                                               data.S[l + m] * data.S[m + n] * data.R[n + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 3
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1
