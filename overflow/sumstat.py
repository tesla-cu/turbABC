import numpy as np
import os


class TruthData(object):
    def __init__(self, valid_folder, case):
        self.sumstat_true = np.empty((0,))
        self.length = []
        if 'cp' in case:
            self.cp = np.loadtxt(os.path.join(valid_folder, 'experiment_cp.txt'))
            self.sumstat_true = np.hstack((self.sumstat_true, self.cp[:, 1]))
            self.length.append(len(self.sumstat_true))
        if 'u' in case:
            self.x, self.u_flat, self.u = self.readfile_with_zones(os.path.join(valid_folder, 'experiment_u.txt'))
            self.sumstat_true = np.hstack((self.sumstat_true, self.u_flat[:, 1]))
            self.length.append(len(self.sumstat_true))
        if 'uv' in case:
            self.x, self.uv_flat, self.uv = self.readfile_with_zones(os.path.join(valid_folder, 'experiment_uv.txt'))
            self.sumstat_true = np.hstack((self.sumstat_true, self.uv_flat[:, 1]))
            self.length.append(len(self.sumstat_true))

    @staticmethod
    def readfile_with_zones(filename):
        with open(filename) as f:
            text = f.readlines()
        x, ind, array = [], [], []
        for line in text:
            if line[:4] == 'ZONE':
                ind.append(len(array))
                x.append(line[-8:-2])
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
        self.grid = self.load_grid(grid_folder)

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
    def load_grid(folder):
        size = np.fromfile(os.path.join(folder, 'size'), dtype=np.int32)
        grid_y = np.fromfile(os.path.join(folder, 'grid_y'))
        grid_y = np.rollaxis(grid_y.reshape((size[2], size[0])), 1)
        grid_x = np.fromfile(os.path.join(folder, 'grid_x'))
        grid_x = np.rollaxis(grid_x.reshape((size[2], size[0])), 1)
        return grid_x, grid_y


def calc_sum_stat(x, y, valid_data_x):
    return np.interp(valid_data_x, x, y)
