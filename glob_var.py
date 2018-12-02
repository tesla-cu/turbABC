import numpy as np

path = None
N_proc = 4
C_limits = np.array([[1.0, 1.8],
                     [0.6, 0.77],
                     [1.7, 2],
                     [1.85, 2.05]])


# algorithm = 'abc'
N = 4
algorithm = 'imcmc'
N_chain = 100
x = 0.05


std = 0.0
eps = 0.0
par_process = None
