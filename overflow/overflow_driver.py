import os
import subprocess
import logging
from scipy.io import FortranFile
import numpy as np


class Overflow(object):
    def __init__(self, job_folder, data_folder, exe_path, MPI_NP):
        self.MPI_NP = MPI_NP
        self.job_folder = job_folder
        self.exe_path = exe_path
        self.env = dict(os.environ)

        with open(os.path.join(data_folder, 'template_over.namelist'), 'r') as file_template:
            self.input_lines = file_template.readlines()

    def write_inputfile(self, c):
        self.input_lines[7] = '    OD_BETAST = {}, OD_BETA1 = {}, OD_BETA2 = {}, OD_SIGW1 = {},\n'.format(c[0], c[1], c[2], c[3])
        self.input_lines[8] = '    OD_A1 = {},\n'.format(c[4])

        with open(os.path.join(self.job_folder, 'over.namelist'), 'w') as f:
            f.writelines(self.input_lines)

    def run_overflow(self, i):
        exe = os.path.join(self.exe_path, 'overflowmpi')
        outfile = os.path.join(self.job_folder, 'over.out')
        # cp = os.path.join(self.basefolder, 'cp')

        # Run overflow
        with open(outfile, 'w', 8) as f:
            logging.info(['mpiexec', '-n', str(self.MPI_NP), exe])
            subprocess.Popen(['mpiexec', '-n', str(self.MPI_NP), exe],
                             cwd=self.job_folder, env=self.env, stdout=f, stderr=f).wait()

        # data = os.path.join(self.basefolder, 'pratio.dat.{}'.format(i))
        # with open(data, 'w', 8) as f:
        #     logging.info([cp])
        #     subprocess.Popen([cp], cwd=self.basefolder, env=self.env, stdout=f).wait()

        return

    @staticmethod
    def read_data_from_overflow(job_folder, grid, indices):
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
        u = np.rollaxis(data[1] / data[0], 1)
        ###################
        surface_data = data[:, 0, :]
        v2 = 0.5 * (surface_data[1] ** 2 + surface_data[2] ** 2 + surface_data[3] ** 2) / surface_data[0]
        p = (surface_data[5] - 1.0) * (surface_data[4] - v2)
        cp = (p - 1.0 / surface_data[5]) / (0.5 * fm * fm)
        ###################
        x_grid, y_grid = grid
        dUdy = np.empty((jd, ld))
        for j in range(jd):
            dUdy[j] = np.gradient(u[j], y_grid[j])
        uv = np.rollaxis(data[6] / data[7], 1) * dUdy
        return cp, u[indices], uv[indices]

