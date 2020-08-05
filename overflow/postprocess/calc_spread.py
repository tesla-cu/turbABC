import os
import numpy as np
from pyabc.distance import calc_err_norm2
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from overflow.sumstat import TruthData, GridData

plt.style.use('dark_background')

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 1 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8 * plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8 * plt.rcParams['font.size']
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


def dist_by_sumstat(sumstat, sumstat_true):
    dist = np.empty(len(sumstat))
    for i, line in enumerate(sumstat):
        dist[i] = calc_err_norm2(line, sumstat_true)
    return dist

basefolder = '../overflow_results/'
nominal_folder = os.path.join(basefolder, 'nominal_data', )
output_folder = os.path.join(basefolder, 'spread', 'calibration_job100', )
plot_folder = os.path.join(basefolder, 'spread', 'plots_spread', )
exp_folder = os.path.join('./', 'valid_data', )
if not os.path.isdir(plot_folder):
    os.makedirs(plot_folder)



Truth = TruthData(exp_folder, ['cp', 'u', 'uv', 'x_separation'])
sumstat_true = Truth.sumstat_true
norm = Truth.norm
print(Truth.length)

# cp_nominal = np.fromfile(os.path.join(nominal_folder, 'cp_all.bin'), dtype=float)
# u_nominal = np.fromfile(os.path.join(nominal_folder, 'u_slice.bin'), dtype=float)
# uv_nominal = np.fromfile(os.path.join(nominal_folder, 'uv_slice.bin'), dtype=float)
#
# cp_maps = np.empty((0, len(cp_nominal)))
# u_maps = np.empty((0, len(u_nominal)))
# uv_maps = np.empty((0, len(u_nominal)))
# x_maps = np.empty((0, 2))
#
# cp_maps = np.vstack((cp_maps, np.fromfile(os.path.join(output_folder, 'cp_all.bin')).reshape(-1, 721)))
# u_maps = np.vstack((u_maps, np.fromfile(os.path.join(output_folder, 'u_slice.bin')).reshape(-1, 800)))
# uv_maps = np.vstack((uv_maps, np.fromfile(os.path.join(output_folder, 'uv_slice.bin')).reshape(-1, 800)))
# with open(os.path.join(output_folder, 'result.dat')) as f:
#     lines = f.readlines()
#     for line in lines:
#         d = np.fromstring(line[1:-1], dtype=float, sep=',')
#         x_maps = np.vstack((x_maps, d[-3:-1].reshape(-1, 2)))

result = np.empty((0, 4 + len(sumstat_true) + 1))
with open(os.path.join(output_folder, 'result.dat')) as f:
    lines = f.readlines()
    for line in lines:
        d = np.fromstring(line[1:-1], dtype=float, sep=',')
        result = np.vstack((result, d))
sumstat = result[:, 4:-1]
print(sumstat.shape)
print('std', np.std(sumstat, axis=0))
print('std dist', np.std(result[:, -1]))