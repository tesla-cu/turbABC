import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use('dark_background')

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 1 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 1
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
# dessertation size
single_column = 235
oneandhalf_column = 352
double_column = 470


def fig_size(width_column):
   fig_width_pt = width_column
   inches_per_pt = 1.0 / 72.27  # Convert pt to inches
   golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
   fig_width = fig_width_pt * inches_per_pt  # width in inches
   fig_height = fig_width * golden_mean  # height in inches
   return fig_width, fig_height

folder = './plots/'
folder_valid = './valid_data/'

strain_true = np.loadtxt(os.path.join(folder_valid, 'ske.txt'))

print(strain_true.shape)
fig_width, fig_height = fig_size(single_column)
fig = plt.figure(figsize=(fig_width, 1.3*fig_height))
ax = plt.gca()
ax.scatter(strain_true[:, 0], strain_true[:, 1])
ax.set_xlabel(r'$t \epsilon_0 / k_0$')
ax.set_ylabel(r'$\overline{S}_{11} k_0/\epsilon_0$')
# ax.axis(xmin=0, xmax=5, ymin=0, ymax=2.5)
fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
fig.savefig(os.path.join(folder, 'strain'))

