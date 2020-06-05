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
        self.input_lines[7] = f'    OD_BETAST = {c[0]}, OD_BETA1 = {c[2]}, OD_BETA2 = {c[3]}, OD_SIGW1 = {c[1]},\n'
        self.input_lines[8] = f'    OD_A1 = {c[4]},\n'
        with open(os.path.join(self.job_folder, 'over.namelist'), 'w') as f:
            f.writelines(self.input_lines)

    def write_debug(self):
        add_debug_line = self.input_lines.copy()
        add_debug_line[1] = '    DEBUG = 1,' + self.input_lines[1][3:]
        with open(os.path.join(self.job_folder, 'over.namelist'), 'w') as f:
            f.writelines(add_debug_line)

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
    def read_data_from_file(filepath):
        f = FortranFile(filepath, 'r')
        f.read_ints(np.int32)[0]  # ng: number of geometries
        (jd, kd, ld, nq, nqc) = tuple(f.read_ints(np.int32))
        type = np.array([np.dtype('<f8')] * 16)
        type[7] = np.dtype('<i4')
        (fm, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = tuple(f.read_record(*type))
        # fm, a, re, t, gamma, beta, tinf, igamma, htinf, ht1, ht2, rgas1, rgas2, refmach, tvref, dtvref))
        data = f.read_reals(dtype=np.float64).reshape((nq, ld, kd, jd))
        return (jd, kd, ld), fm, data[:, :, 1, :]  # taking 2D data (middle in y direction)

    @staticmethod
    def calc_Cp(surface_data, mach_number):
        v2 = 0.5 * (surface_data[1] ** 2 + surface_data[2] ** 2 + surface_data[3] ** 2) / surface_data[0]
        p = (surface_data[5] - 1.0) * (surface_data[4] - v2)
        return (p - 1.0 / surface_data[5]) / (0.5 * mach_number * mach_number)

    @staticmethod
    def calc_separation_point(x_grid, u_surface):
        # separation point
        index_negative = np.where(u_surface < 0)[0]
        if len(index_negative) > 0:
            x_separation = x_grid[index_negative[-1], 0]  # x stored in inverse order
            x_reattachment = x_grid[index_negative[0], 0]
        else:
            x_separation = x_grid[0, 0]  # x stored in inverse order
            x_reattachment = x_grid[0, 0]
        # np.savetxt('./u_surface', [x_grid[:, 0], u_surface])
        return np.array([x_separation, x_reattachment])

    @staticmethod
    def read_data_from_overflow(job_folder, grid, x_slices, y_slices):

        #read 2D data from file
        (jd, kd, ld), mach_number, data = Overflow.read_data_from_file(os.path.join(job_folder, 'q.restart'))
        u = np.rollaxis(data[1] / data[0] / mach_number, 1)
        x_grid, y_grid = grid
        ###################
        cp = Overflow.calc_Cp(data[:, 0, :], mach_number)
        ###################
        x_separation = Overflow.calc_separation_point(x_grid=x_grid, u_surface=u[:, 1])
        # u profiles

        y_grid = y_grid - y_grid[:, 0].reshape(-1, 1)
        xy_grid = np.hstack((x_grid.flatten().reshape(-1, 1), y_grid.flatten().reshape(-1, 1)))
        # interpolate in experimental points only
        u_interp = interpolate.griddata(xy_grid, u.flatten(), x_slices, method='cubic')
        # interpolate in the x slice, where experiment is taken
        u_interp_slice = interpolate.griddata(xy_grid, u.flatten(), y_slices, method='cubic')

        # u'v' profiles
        dUdy = np.empty((jd, ld))
        for j in range(jd):
            dUdy[j] = np.gradient(u[j], y_grid[j])
        mu_t = Overflow.read_data_from_file(os.path.join(job_folder, 'q.turb'))[2][3]       # data is 3rd in returned tuple and mu_t is 4th in data
        uv = np.rollaxis(mu_t/data[0], 1)*dUdy / 2.763e6   # normalized by Re = 2.763e6
        # uv = np.rollaxis(data[6] / data[7], 1) * dUdy     # approximate as k/w * dUdy

        # interpolate in experimental points only
        uv_interp = interpolate.griddata(xy_grid, uv.flatten(), x_slices, method='cubic')
        # interpolate in the x slice, where experiment is taken
        uv_interp_slice = interpolate.griddata(xy_grid, uv.flatten(), y_slices, method='cubic')

        return cp, u_interp, uv_interp, u_interp_slice, uv_interp_slice, u[:, 1], x_separation
