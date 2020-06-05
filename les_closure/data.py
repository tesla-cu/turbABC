import logging
import os
import numpy as np
from numpy.fft import fftfreq, fftn, ifftn
from numpy.linalg import norm
from time import time

from pyabc.utils import timer


class Data(object):
    def __init__(self, data_dict, delta, dx):

        self.delta = delta
        self.S = calc_strain_tensor(data_dict, dx)
        self.R = calc_rotation_tensor(data_dict, dx)


class DataFiltered(object):
    def __init__(self, valid_folder, case_params):

        dx = np.array([case_params['lx'] / case_params['N_point']] * 3)
        HIT_data = load_HIT_data(valid_folder, case_params['N_point'])
        logging.info('Filter HIT data')
        LES_data = self.filter3d_dict(data=HIT_data, scale_k=case_params['LES_scale'], dx=dx, )
        del HIT_data
        logging.info('Writing file')
        np.savez(os.path.join(valid_folder, 'LES.npz'), **LES_data)
        logging.info('Create LES class')
        self.delta = 1 / case_params['LES_scale']
        LES = Data(LES_data, self.delta, dx)
        del LES_data
        self.S = LES.S
        self.R = LES.R

    @staticmethod
    def tophat_kernel_3d(k, limit):
        """ Create kernel (3D array) of Tophat filter (low-pass filter) in Fourier space.
            k - array of wave numbers;
            limit - cutoff wavenumber."""
        a = np.array([[[np.sqrt(kx ** 2 + ky ** 2 + kz ** 2) for kz in k[2]] for ky in k[1]] for kx in k[0]])
        kernel = (a <= limit)
        # kernel = np.piecewise(a, [a <= limit, a > limit], [1, 0])
        return kernel

    @staticmethod
    def sinc_kernel_3d(k, limit):
        """ Create 3D array of sinc(s) filter (sharp filter in physical space)
        :param k: array of wave numbers;
        :param limit:
        :return: kernel array
        """
        a = np.array([[[np.sqrt(kx ** 2 + ky ** 2 + kz ** 2) for kz in k[2]] for ky in k[1]] for kx in k[0]])
        kernel = np.sinc(a / limit)
        return kernel

    @staticmethod
    def filter3d_array(data, kernel):
        """ Filtering of ndarray (performed as multiplication in Fourier space)
        :param data: nD np.array of data
        :param kernel: nD np.array of filtering kernel
        :return: filtered nD np.array
        """
        fft_array = fftn(data)                            # FFT
        fft_filtered = np.multiply(fft_array, kernel)     # Filtering
        return ifftn(fft_filtered).real                   # iFFT

    @staticmethod
    def filter3d_dict(data, scale_k, dx, filename=None):
        """ Tophat (low-pass) filtering for dictionary of 3D arrays
            (performed as multiplication in Fourier space)
        :param data: dict of 3d np.arrays
        :param scale_k: wave number, which define size of filter
        :param dx: distance between data points in physical space
        :param filename: filename if need to save filtered result in .npz file
        :return: dict of filtered arrays
        """
        start = time()
        N_points = next(iter(data.values())).shape  # shape of any array in dict (supposed to be the same shapes)
        k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
        kernel = DataFiltered.tophat_kernel_3d(k, scale_k)              # Create filter kernel

        result = dict()
        for key, value in data.items():
            result[key] = DataFiltered.filter3d_array(value, kernel)
        end = time()
        print(end - start)
        timer(start, end, 'Time for data filtering')

        if filename:
            logging.info('\nWrite file in ./data/' + filename + '.npz')
            file = './data/' + filename + '.npz'
            np.savez(file, **result)

        return result


def field_gradient(field, dx):
    """Calculate tensor of gradients of self.field.
    :return:      dictionary of gradient tensor
    """
    grad = dict()
    grad['uu'], grad['uv'], grad['uw'] = np.gradient(field['u'], dx[0], dx[1], dx[2])
    grad['vu'], grad['vv'], grad['vw'] = np.gradient(field['v'], dx[0], dx[1], dx[2])
    grad['wu'], grad['wv'], grad['ww'] = np.gradient(field['w'], dx[0], dx[1], dx[2])
    return grad


def calc_strain_tensor(field, dx):
    """Calculate strain tensor S_ij = 1/2(du_i/dx_j+du_j/dx_i) of given field.
    :return:      dictionary of strain tensor
    """
    A = field_gradient(field, dx)
    tensor = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
    return tensor


def calc_rotation_tensor(field, dx):
    """Calculate rotation tensor R_ij = 1/2(du_i/dx_j-du_j/dx_i) of given field.
    :return:       dictionary of rotation tensor
    """
    A = field_gradient(field, dx)
    tensor = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor[i + j] = 0.5 * (A[i + j] - A[j + i])
    return tensor


def load_HIT_data(valid_folder, N_points):
    datafile = dict()
    # 'JHU_data':
    datafile['u'] = os.path.join(valid_folder, 'HIT_u.bin')
    datafile['v'] = os.path.join(valid_folder, 'HIT_v.bin')
    datafile['w'] = os.path.join(valid_folder, 'HIT_w.bin')
    type_of_bin_data = np.float32

    logging.info('Load HIT data')
    HIT_data = dict()
    data_shape = ([N_points, N_points, N_points])
    for i in ['u', 'v', 'w']:
        HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
    for key, value in HIT_data.items():
        HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])
    return HIT_data