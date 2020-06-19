import numpy as np
import os


class TruthData(object):
    def __init__(self, valid_folder, case):
        self.sumstat_true = np.empty((0,))
        self.length = []
        self.norm = np.empty((0,))
        if 'cp' in case:
            self.cp = np.loadtxt(os.path.join(valid_folder, 'experiment_cp.txt'))
            number_of_datapoints = len(self.cp[:, 1])
            norm = np.max(np.abs(self.cp[:, 1])) * np.ones(number_of_datapoints)
            norm *= np.sqrt(number_of_datapoints)      # !!! use only with norm2
            self.norm = np.hstack((self.norm, norm))
            self.sumstat_true = np.hstack((self.sumstat_true, self.cp[:, 1]))
            self.length.append(len(self.sumstat_true))
        if 'u' in case:
            self.x, self.u_flat, self.u = readfile_with_zones(os.path.join(valid_folder, 'experiment_u.txt'))
            number_of_datapoints = len(self.u_flat[:, 0])
            norm = np.max(np.abs(self.u_flat[:, 0]))*np.ones(number_of_datapoints)
            norm *= np.sqrt(number_of_datapoints)      # !!! use only with norm2
            self.norm = np.hstack((self.norm, norm))
            self.sumstat_true = np.hstack((self.sumstat_true, self.u_flat[:, 0]))
            self.length.append(len(self.sumstat_true))
        if 'uv' in case:
            self.x, self.uv_flat, self.uv = readfile_with_zones(os.path.join(valid_folder, 'experiment_uv.txt'))
            number_of_datapoints = len(self.uv_flat[:, 0])
            norm = np.max(np.abs(self.uv_flat[:, 0]))*np.ones(number_of_datapoints)
            norm *= np.sqrt(number_of_datapoints)      # !!! use only with norm2
            self.norm = np.hstack((self.norm, norm))
            self.sumstat_true = np.hstack((self.sumstat_true, -self.uv_flat[:, 0]))
            self.length.append(len(self.sumstat_true))
        if 'x_separation' in case:
            self.sumstat_true = np.hstack((self.sumstat_true, [0.7, 1.1]))
            self.norm = np.hstack((self.norm, [1, 1]))
            self.length.append(len(self.sumstat_true))
        # print('lenth', np.diff(self.length))
        self.sumstat_true /= self.norm


def readfile_with_zones(filename):
    with open(filename) as f:
        text = f.readlines()
    x, ind, array = [], [], []
    for line in text:
        if line[:4] == 'ZONE':
            ind.append(len(array))
            x.append(float(line[-8:-2]))
        else:
            array.append([float(i) for i in line.split()])
    ind.append(len(array))
    experiment = [np.array(array[ind[i]:ind[i + 1]]) for i in range(len(ind) - 1)]
    return x, np.array(array), experiment

    # @staticmethod
    # def flat_to_lists(array, ind):
    #     return [array[ind[i]:ind[i + 1]] for i in range(len(ind) - 1)]


class GridData(object):
    def __init__(self, grid_folder):
        self.grid_y = self.load_grid_y(grid_folder)
        self.grid_x = self.load_grid_x(grid_folder)
        self.indices = self.load_indices(grid_folder)
        self.x_slices = self.load_x_slices(grid_folder)
        self.grid = self.load_grid(grid_folder)
        self.xy_flat_grid = self.flatten_grid()
        self.y_slices = self.make_y_slices(grid_folder)


    @staticmethod
    def load_grid_y(folder):
        size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
        grid_y = np.fromfile(os.path.join(folder, 'grid_y'))
        grid_y = np.rollaxis(grid_y.reshape((size[2], size[0])), 1)
        indices = np.flip(np.loadtxt(os.path.join(folder, 'indices'), dtype=np.int)) - 1
        grid_y = grid_y[indices, :]
        grid_y = grid_y - grid_y[:, 0].reshape(-1, 1)
        return grid_y

    @staticmethod
    def load_grid_x(folder):
        size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
        grid_x = np.fromfile(os.path.join(folder, 'grid_x'))
        grid_x = np.rollaxis(grid_x.reshape((size[2], size[0])), 1)
        return grid_x[:, 0]

    @staticmethod
    def load_indices(folder):
        return np.flip(np.loadtxt(os.path.join(folder, 'indices'), dtype=np.int)) - 1

    @staticmethod
    def load_x_slices(folder):
        x, u_flat, u = readfile_with_zones(os.path.join(folder, 'experiment_uv.txt'))
        x_slices = np.empty((0, 2))
        for i, u_x in enumerate(u):
            i_slices = np.hstack((np.repeat(x[i], len(u_x)).reshape((-1, 1)), u_x[:, 1].reshape((-1, 1))))
            x_slices = np.vstack((x_slices, i_slices))
        return x_slices

    @staticmethod
    def make_y_slices(folder):
        n = 100
        x_experiment, _, _ = readfile_with_zones(os.path.join(folder, 'experiment_u.txt'))
        y = np.linspace(0, 0.14, n)
        y_slices = np.empty((len(x_experiment)*n, 2))
        for i, x in enumerate(x_experiment):
            y_slices[i*n:(i+1)*n, 0] = x*np.ones(n)
            y_slices[i * n:(i + 1) * n, 1] = y
        return y_slices

    def flatten_grid(self):
        x_grid, y_grid = self.grid
        return np.hstack((x_grid.flatten().reshape(-1, 1), y_grid.flatten().reshape(-1, 1)))

    @staticmethod
    def load_grid(folder):
        size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
        grid_y = np.fromfile(os.path.join(folder, 'grid_y'))
        grid_y = np.rollaxis(grid_y.reshape((size[2], size[0])), 1)
        grid_x = np.fromfile(os.path.join(folder, 'grid_x'))
        grid_x = np.rollaxis(grid_x.reshape((size[2], size[0])), 1)
        return grid_x, grid_y


def calc_sum_stat(x, y, valid_data_x):
    return np.interp(valid_data_x, x, y)
