import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyabc.abc_alg as m
import pyabc.utils as utils
import rans_ode as rans
from tmp.odeSolveMethods import RungeKuttaFehlberg as RK

# plt.style.use('dark_background')

fig_width_pt = 1.5*246.0  # Get this from LaTeX using "The column width is: \the\columnwidth \\"
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']-1
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

folder = './plots/'

basefolder = './'
path = {'output': os.path.join(basefolder, 'output'), 'plots': os.path.join(basefolder, 'plots/')}
if not os.path.isdir(path['plots']):
    os.makedirs(path['plots'])

folder_data = './valid_data/'

##########################################################
# Decay
##########################################################
decay_exp = np.loadtxt(os.path.join(folder_data, 'decay_exp_a.txt'))
c = [1.5, 0, 1.44, 1.92]

tspan = [0, 45]
a110 = 0.36
a220 = -0.08
a330 = -0.28
T_lrr, Y_lrr = RK(f=rans.rans_decay, tspan=tspan, u0=[1, 1, a110, a220, a330, 0, 0, 0], t_step=0.001, args=c)
tau_lrr = (1/(2*(c[3]-1)))*np.log(1+(c[3]-1)*T_lrr)

fig = plt.figure(figsize=(0.8*fig_width, 1.3*fig_height))
ax = plt.gca()
ax.scatter(decay_exp[:, 0], 2*decay_exp[:, 1], label='exp')
plt.plot(tau_lrr, Y_lrr[:, 2], 'm--', label='a11')
plt.plot(tau_lrr, Y_lrr[:, 3], 'm--', label='a22')
plt.plot(tau_lrr, Y_lrr[:, 4], 'm--', label='a33')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$a$')
ax.axis(xmin=0, xmax=0.5, ymin=-0.4, ymax=0.4)
plt.legend(loc=0)
fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.95)
fig.savefig(os.path.join(folder, 'decay_experiment'))
plt.close('all')

##########################################################
# Strain-relax
##########################################################
k0e0 = 0.0092/0.0035  #initial value of k/e
#Set up straining
a1 = 9*k0e0**2
a2 = 10*k0e0**2
a3 = 18*k0e0**2
a4 = 8*k0e0**2
S0 = 0
Lt = 0.1/k0e0
t1 = 0.25
t3 = 0.55
t2 = (a1*t1+a2*t3)/(a1+a2)
t4 = 0.70
t6 = 0.95
t5 = (a3*t4+a4*t6)/(a3+a4)
strain_params = [a1, a2, a3, a4, S0, Lt, t1, t2, t3, t4, t5, t6]

tspan = [0, 0.95]
t_array = np.linspace(0, 0.95, 1000)
S = np.empty_like(t_array)
for i, t in enumerate(t_array):
    S[i] = utils.strain_tensor(t, strain_params)

ske_exp = np.loadtxt(os.path.join(folder_data, 'ske.txt')) #experimental straining from Chen et al.
fig = plt.figure(figsize=(fig_width, fig_height))
ax = plt.gca()
ax.scatter(ske_exp[:, 0], ske_exp[:, 1], label='experiment')
plt.plot(t_array, S, 'm--', label='approximation')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\frac{\overline{S}_{11}k_0}{\epsilon_0}$')
ax.axis(xmin=0, xmax=1, ymin=-10, ymax=10)
plt.legend(loc=0)
fig.subplots_adjust(left=0.17, right=0.95, bottom=0.14, top=0.95)
fig.savefig(os.path.join(folder, 'ske'))
plt.close('all')

c = [1.5, 0.8, 1.44, 1.92]
T_lrr, Y_lrr = RK(f=rans.rans_strain_relax, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0], t_step=0.0001, args=(c, strain_params))
strain_relax_exp = np.loadtxt(os.path.join(folder_data, 'strain_relax_b11.txt'))

fig = plt.figure(figsize=(fig_width, fig_height))
ax = plt.gca()
ax.scatter(strain_relax_exp[:, 0], strain_relax_exp[:, 1], label='exp')
plt.plot(T_lrr, Y_lrr[:, 2], 'm--', label='a11')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$a$')
ax.axis(xmin=0, xmax=1, ymin=-1, ymax=1)
plt.legend(loc=0)
fig.subplots_adjust(left=0.15, right=0.95, bottom=0.14, top=0.95)
fig.savefig(os.path.join(folder, 'strain_relax_experiment'))
plt.close('all')