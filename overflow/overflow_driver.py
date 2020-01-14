import os
import subprocess as sp
import logging
from scipy.io import FortranFile
from scipy import interpolate
import numpy as np
from time import time
from pyabc.utils import timer


class Overflow(object):
    def __init__(self, job_folder, data_folder, exe_path, MPI_NP):
        self.MPI_NP = MPI_NP
        self.job_folder = job_folder
        self.exe_path = exe_path
        self.env = dict(os.environ)

        with open(os.path.join(data_folder, 'template_over.namelist'), 'r') as file_template:
            self.input_lines = file_template.readlines()

    def write_inputfile(self, c):
        self.input_lines[7] = '    OD_BETAST = {}, OD_BETA1 = {}, OD_BETA2 = {}, OD_SIGW1 = {},\n'.format(c[0], c[2], c[3], c[1])
        self.input_lines[8] = '    OD_A1 = {},\n'.format(c[4])
        with open(os.path.join(self.job_folder, 'over.namelist'), 'w') as f:
            f.writelines(self.input_lines)

    def run_overflow(self, i):
        exe = os.path.join(self.exe_path, 'a.out')
        outfile = os.path.join(self.job_folder, 'over.out')
        # Run overflow
        time_start = time()
        args = ['mpiexec', '-np', str(self.MPI_NP), '-d', self.job_folder, exe]
        logging.info(args)
        with open(outfile, 'wb', 8) as f:
            sp.Popen(args, cwd=self.job_folder, env=self.env, stdout=f, stderr=f).wait()
        time_end = time()
        timer(time_start, time_end, 'Overflow time')
        return

    @staticmethod
    def read_data_from_overflow(job_folder, grid, x_slices, y_slices):
        ########################################################################
        # Read data
        ########################################################################
        f = FortranFile(os.path.join(job_folder, 'q.save'), 'r')
        f.read_ints(np.int32)[0]  # ng: number of geometries
        (jd, kd, ld, nq, nqc) = tuple(f.read_ints(np.int32))
        type = np.array([np.dtype('<f8')] * 16)
        type[7] = np.dtype('<i4')
        (fm, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = tuple(f.read_record(*type))
        # fm, a, re, t, gamma, beta, tinf, igamma, htinf, ht1, ht2, rgas1, rgas2, refmach, tvref, dtvref))
        data = f.read_reals(dtype=np.float64).reshape((nq, ld, kd, jd))
        data = data[:, :, 1, :]  # taking 2D data (middle in y direction)
        ###################
        u = np.rollaxis(data[1] / data[0] / fm, 1)

        v = np.rollaxis(data[2] / data[0] / fm, 1)
        ###################
        surface_data = data[:, 0, :]
        v2 = 0.5 * (surface_data[1] ** 2 + surface_data[2] ** 2 + surface_data[3] ** 2) / surface_data[0]
        p = (surface_data[5] - 1.0) * (surface_data[4] - v2)
        cp = (p - 1.0 / surface_data[5]) / (0.5 * fm * fm)
        ###################
        x_grid, y_grid = grid

        # x_grid
        dUdy = np.empty((jd, ld))
        dVdx = np.empty((jd, ld))
        for j in range(jd):
            dUdy[j] = np.gradient(u[j], y_grid[j])
        for l in range(ld):
            dVdx[:, l] = np.gradient(v[:, l], x_grid[:, l])
        uv = np.rollaxis(data[6] / data[7], 1) * (dUdy + dVdx)

        y_grid = y_grid - y_grid[:, 0].reshape(-1, 1)
        xy_grid = np.hstack((x_grid.flatten().reshape(-1, 1), y_grid.flatten().reshape(-1, 1)))
        # interpolate in experimental points only
        u_interp = interpolate.griddata(xy_grid, u.flatten(), x_slices, method='cubic')
        uv_interp = interpolate.griddata(xy_grid, uv.flatten(), x_slices, method='cubic')
        # interpolate in the x slice, where experiment is taken
        u_interp_slice = interpolate.griddata(xy_grid, u.flatten(), y_slices, method='cubic')
        uv_interp_slice = interpolate.griddata(xy_grid, uv.flatten(), y_slices, method='cubic')
        return cp, u_interp, uv_interp, u_interp_slice, uv_interp_slice
        # return cp, u_interp[indices], uv_interp[indices]

    @staticmethod
    def calc_mut_from_overflow(job_folder, grid):
        ########################################################################
        # Read data
        ########################################################################
        f = FortranFile(os.path.join(job_folder, 'q.save'), 'r')
        f.read_ints(np.int32)[0]  # ng: number of geometries
        (jd, kd, ld, nq, nqc) = tuple(f.read_ints(np.int32))
        type = np.array([np.dtype('<f8')] * 16)
        type[7] = np.dtype('<i4')
        (fm, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = tuple(f.read_record(*type))
        # fm, a, re, t, gamma, beta, tinf, igamma, htinf, ht1, ht2, rgas1, rgas2, refmach, tvref, dtvref))
        data = f.read_reals(dtype=np.float64).reshape((nq, ld, kd, jd))
        data = data[:, :, 1, :]  # taking 2D data (middle in y direction)
        ###################
        u = np.rollaxis(data[1] / data[0] / fm, 1)
        v = np.rollaxis(data[2] / data[0] / fm, 1)
        ###################
        x_grid, y_grid = grid
        dUdy = np.empty((jd, ld))
        for j in range(jd):
            dUdy[j] = np.gradient(u[j], y_grid[j])
        omega = np.rollaxis(data[7], 1)
        print(dUdy.shape, omega.shape, y_grid.shape)
        print(np.sum(dUdy > 0.31*omega))
        rows, cols = np.where(dUdy > 0.31*omega)
        print(rows.shape, cols.shape)
        print(rows)
        print(y_grid[np.where(dUdy > 0.31*omega)].shape)

        # dVdx = np.empty((jd, ld))
        # for l in range(ld):
        #     dVdx[:, l] = np.gradient(v[:, l], x_grid[:, l])
        # uv = np.rollaxis(data[6] / data[7], 1) * (dUdy + dVdx)

        # y_grid = y_grid - y_grid[:, 0].reshape(-1, 1)
        # xy_grid = np.hstack((x_grid.flatten().reshape(-1, 1), y_grid.flatten().reshape(-1, 1)))
        # # interpolate in experimental points only
        # u_interp = interpolate.griddata(xy_grid, u.flatten(), x_slices, method='cubic')
        # uv_interp = interpolate.griddata(xy_grid, uv.flatten(), x_slices, method='cubic')
        # # interpolate in the x slice, where experiment is taken
        # u_interp_slice = interpolate.griddata(xy_grid, u.flatten(), y_slices, method='cubic')
        # uv_interp_slice = interpolate.griddata(xy_grid, uv.flatten(), y_slices, method='cubic')
        return

