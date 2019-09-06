import os
from scipy.integrate import odeint
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyabc.abc_alg as m
import pyabc.utils as utils
import rans_ode.ode as rans
import pyabc.glob_var as g
import rans_ode.sumstat as sumstat
import rans_ode.workfunc_rans as workfunc

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


folder_valid = '../rans_ode/valid_data/'


g.Strain = workfunc.StrainTensor(valid_folder=folder_valid)


def plot_impulsive(c, plot_folder):
    g.Truth = sumstat.TruthData(valid_folder=folder_valid, case='impulsive')
    u0 = [1, 1, 0, 0, 0, 0, 0, 0]
    # axisymmetric expansion
    tspan1 = np.linspace(0, 1.5 / np.abs(g.Strain.axi_exp[0]), 200)
    Ynke1 = odeint(rans.rans_impulsive, u0, tspan1, args=(c, g.Strain.axi_exp), atol=1e-8, mxstep=200)
    # axisymmetric contraction
    tspan2 = np.linspace(0, 1.5 / np.abs(g.Strain.axi_con[0]), 200)
    Ynke2 = odeint(rans.rans_impulsive, u0, tspan2, args=(c, g.Strain.axi_con), atol=1e-8, mxstep=200)
    # pure shear
    tspan3 = np.linspace(0, 5 / (2 * g.Strain.pure_shear[3]), 200)
    Ynke3 = odeint(rans.rans_impulsive, u0, tspan3, args=(c, g.Strain.pure_shear), atol=1e-8, mxstep=200)
    # plane strain
    tspan4 = np.linspace(0, 1.5 / g.Strain.plane_strain[0], 200)
    Ynke4 = odeint(rans.rans_impulsive, u0, tspan4, args=(c, g.Strain.plane_strain), atol=1e-8, mxstep=200)

    fig = plt.figure(figsize=(0.8 * fig_width, 1.3 * fig_height))
    ax = plt.gca()

    ax.plot(np.abs(g.Strain.axi_exp[0]) * tspan1, Ynke1[:, 0], label='axisymmetric expansion')
    ax.scatter(g.Truth.axi_exp_k[:, 0], g.Truth.axi_exp_k[:, 1], marker='o')

    ax.plot(np.abs(g.Strain.axi_con[0]) * tspan2, Ynke2[:, 0], label='axisymmetric contraction')
    ax.scatter(g.Truth.axi_con_k[:, 0], g.Truth.axi_con_k[:, 1], marker='o')

    ax.plot(2 * g.Strain.pure_shear[3] * tspan3, Ynke3[:, 0], label='pure shear')
    ax.scatter(g.Truth.shear_k[:, 0], g.Truth.shear_k[:, 1], marker='o')

    ax.plot(np.abs(g.Strain.plane_strain[0]) * tspan4, Ynke4[:, 0], label='plain strain')
    ax.scatter(g.Truth.plane_k[:, 0], g.Truth.plane_k[:, 1], marker='o')

    ax.set_xlabel(r'$S\cdot t$')
    ax.set_ylabel(r'$k/k_0$')
    # ax.axis(xmin=0, xmax=5, ymin=0, ymax=2.5)
    plt.legend()
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(plot_folder, 'compare_impulsive_k'))

    fig = plt.figure(figsize=(0.8 * fig_width, 1.3 * fig_height))
    ax = plt.gca()

    ax.plot(np.abs(g.Strain.axi_exp[0]) * tspan1, Ynke1[:, 2], label='axisymmetric expansion')
    ax.scatter(g.Truth.axi_exp_a[:, 0], 2*g.Truth.axi_exp_a[:, 1], marker='o')

    ax.plot(np.abs(g.Strain.axi_con[0]) * tspan2, Ynke2[:, 2], label='axisymmetric contraction')
    ax.scatter(g.Truth.axi_con_a[:, 0], 2*g.Truth.axi_con_a[:, 1], marker='o')

    ax.plot(np.abs(g.Strain.plane_strain[0]) * tspan4, Ynke4[:, 2], label='plain strain')
    ax.scatter(g.Truth.plane_a[:, 0], 2*g.Truth.plane_a[:, 1], marker='o')

    ax.set_xlabel(r'$S\cdot t$')
    ax.set_ylabel(r'$a_{11}$')
    # ax.axis(xmin=0, xmax=1.5, ymin=0, ymax=2.5)
    plt.legend()
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(plot_folder, 'compare_impulsive_a'))
    plt.close('all')


def plot_periodic(c, plot_folder):
    g.Truth = sumstat.TruthData(valid_folder=folder_valid, case='periodic')
    s0 = 3.3
    beta = [0.125, 0.25, 0.5, 0.75, 1]
    u0 = [1, 1, 0, 0, 0, 0, 0, 0]
    # Periodic shear(five different frequencies)
    tspan = np.linspace(0, 50 / s0, 500)
    fig = plt.figure(figsize=(1 * fig_width, 1.3 * fig_height))
    ax = plt.gca()
    for i in range(5):
        Ynke = odeint(rans.rans_periodic, u0, tspan, args=(c, s0, beta[i]), atol=1e-8, mxstep=200)
        ax.semilogy(s0*tspan, Ynke[:, 0], label=r'$\omega/S_{max} = $' + ' {}'.format(beta[i]))
        ax.scatter(g.Truth.periodic_k[i][:, 0], g.Truth.periodic_k[i][:, 1], marker='o')

    ax.set_xlabel(r'$S\cdot t$')
    ax.set_ylabel(r'$k/k_0$')
    ax.axis(xmin=0, ymin=0, xmax=51)
    plt.legend(loc=2, labelspacing=0.2, borderpad=0.0)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(plot_folder, 'compare_periodic'))

    plt.close('all')


##########################################################
# Decay
##########################################################
def plot_decay(c, plot_folder):

    g.Truth = sumstat.TruthData(valid_folder=folder_valid, case='decay')
    u0 = [1, 1, 0.36, -0.08, -0.28, 0, 0, 0]
    tspan = np.linspace(0, 0.3, 200)
    Ynke = odeint(rans.rans_decay, u0, tspan, args=(c,), atol=1e-8, mxstep=200)
    # tau_lrr = (1/(2*(c[3]-1)))*np.log(1+(c[3]-1)*tspan)

    fig = plt.figure(figsize=(0.8*fig_width, 1.3*fig_height))
    ax = plt.gca()
    ax.scatter(g.Truth.decay_a11[:, 0], 2 * g.Truth.decay_a11[:, 1], color='b', label='exp')
    ax.scatter(g.Truth.decay_a22[:, 0], 2 * g.Truth.decay_a22[:, 1], color='b')
    ax.scatter(g.Truth.decay_a22[:, 0], 2 * g.Truth.decay_a33[:, 1], color='b')
    plt.plot(tspan, Ynke[:, 2], 'm--', label='a11')
    plt.plot(tspan, Ynke[:, 3], 'm--', label='a22')
    plt.plot(tspan, Ynke[:, 4], 'm--', label='a33')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$a$')
    # ax.axis(xmin=0, xmax=0.5, ymin=-0.4, ymax=0.4)
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(plot_folder, 'compare_decay'))
    plt.close('all')


# ##########################################################
# # Strain-relax
# ##########################################################
def plot_strained(c, plot_folder):

    g.Truth = sumstat.TruthData(valid_folder=folder_valid, case='strain-relax')

    u0 = [1, 1, 0.36, -0.08, -0.28, 0, 0, 0]
    # strain-relaxation
    tspan = np.linspace(0.0775, 0.953, 500)
    Ynke = odeint(rans.rans_strain_relax, u0, tspan, args=(c,), atol=1e-8, mxstep=200)

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.scatter(g.Truth.strain_relax_a11[:, 0], g.Truth.strain_relax_a11[:, 1], label='exp')
    plt.plot(tspan, Ynke[:, 2], 'm--', label='a11')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$a$')
    ax.axis(xmin=0, xmax=1, ymin=-1, ymax=1)
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.14, top=0.95)
    fig.savefig(os.path.join(plot_folder, 'compare_strained'))
    plt.close('all')
