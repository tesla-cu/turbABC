
TINY_log = -8
TINY = 10e-8

path = None
C_limits = None
eps = 0.0
std = 0.0
t0 = 0

Truth = None
case = None
prior_interpolator = None
target_acceptance = None

par_process = None  # empty global variable to fill with parallel class in main_rans.py
norm_order = 2               # options: 1 - max, 2 - second norm
work_function = None

# rans ode
Strain = None

# overflow
Grid = None
job_folder = None

# les model
SumStat = None
LesModel = None