import numpy as np

path = None
N_proc = 4
C_limits = np.array([[1.0, 2.8],
                     [0.35, 1.3],
                     [1.35, 2.2],
                     [1.5, 2.5]])


algorithm = 'abc'
N = 10
# algorithm = 'imcmc'
N_chain = 10
x = 0.05


std = 0.0
eps = 0.0
par_process = None
