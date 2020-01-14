import os
import numpy as np

from overflow.overflow_driver import Overflow
from overflow.sumstat import GridData
# import matplotlib as mpl
# mpl.use('pdf')
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
# from cycler import cycler
# import matplotlib.ticker as ticker
# from matplotlib.lines import Line2D
# import matplotlib.colors as colors
# from overflow.sumstat import TruthData, GridData
# plt.style.use('dark_background')

nominal_folder = '../nominal/nominal_data/'
exp_folder = '../overflow/valid_data/'
Grid = GridData(exp_folder)

Overflow.calc_mut_from_overflow(nominal_folder, Grid.grid)