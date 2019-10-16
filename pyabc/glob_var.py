
TINY_log = -8
TINY = 10e-8

path = None
C_limits = None
eps = 0.0
std = 0.0
t0 = 0
work_function = None
Truth = None
case = None
prior_interpolator = None
target_acceptance = None

norm_order = 2               # options: 1 - max, 2 - second norm

# rans ode
par_process = None  # empty global variable to fill with parallel class in main_rans.py
Strain = None

# overflow
Grid = None
job_folder = None