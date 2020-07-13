import logging
import numpy as np
import os
from overflow.sumstat import TruthData
from pyabc.distance import calc_err_norm2
import postprocess.postprocess_func as pp
from pyabc.kde import find_MAP_kde, gaussian_kde_scipy


def dist_by_sumstat(sumstat, sumstat_true):
    dist = np.empty(len(sumstat))
    for i, line in enumerate(sumstat):
        dist[i] = calc_err_norm2(line, sumstat_true)
    return dist


basefolder = '../'
### Paths
path = {'output': os.path.join(basefolder, 'overflow_results/output_4/'),
        'valid_data': '../overflow/valid_data/'}
print('Path:', path)
chains_folder = os.path.join(path['output'], 'for_chains_folder')
if not os.path.isdir(chains_folder):
    os.makedirs(chains_folder)
logging.basicConfig(
    format="%(levelname)s: %(name)s:  %(message)s",
    handlers=[logging.FileHandler(os.path.join(chains_folder, 'prepare_chains.log')), logging.StreamHandler()],
    level=logging.DEBUG)


logging.info('\n############# POSTPROCESSING ############')
data_file = 'joined_data.npz'


x = 0.03
lim = 0.25
N_chains = 200
num_bin_kde = 15
num_bin_raw = [12]*4
mirror = True
chain_with_limits = False


job_folders = [os.path.join(chains_folder, f'chain_{n_job}') for n_job in range(N_chains)]
for i in range(N_chains):
    if not os.path.isdir(job_folders[i]):
        os.makedirs(job_folders[i])


Truth = TruthData(path['valid_data'], ['cp', 'u', 'uv', 'x_separation'])
sumstat_true = Truth.sumstat_true
norm = Truth.norm
print('statistics length:', Truth.length)

logging.info('Loading data from .npz')
c_array = np.load(os.path.join(path['output'], data_file))['c_array']
sumstat_all = np.load(os.path.join(path['output'], data_file))['sumstat_all']
C_limits = np.load(os.path.join(path['output'], data_file))['C_limits']
N_total, N_params = c_array.shape
logging.info(f'There are {N_total} samples total in {N_params}D space')
# !!! stored summary statistics are not divided by norm
####################################################################
print('x1+x2 statistics')
dist_x = dist_by_sumstat(sumstat_all[:, Truth.length[2]:], sumstat_true[Truth.length[2]:])
ind_nonzero = np.where(dist_x < lim)[0]
c_array_x = c_array[ind_nonzero]
print('cp + U + uv statistics')
dist_all = dist_by_sumstat(sumstat_all[ind_nonzero, :Truth.length[2]] / norm[:Truth.length[2]],
                           sumstat_true[:Truth.length[2]])
N_x = len(ind_nonzero)
# accept x percent
ind = np.argsort(dist_all)
accepted = c_array_x[ind]
logging.info('##################################################')
logging.info(f'There are {len(ind)} samples in {N_params}D space')
dist = dist_all[ind]
logging.info('min dist = {} at {}'.format(dist[0], accepted[0]))
n = int(x * N_x)
logging.info(f'x_{x * 100}: {n} samples accepted')
accepted = accepted[:n]
# define eps for chains
eps = dist[n-1]
for n_job in job_folders:
    np.savetxt(os.path.join(n_job, 'eps'), [eps])
logging.info('x = {}, eps = {}, N accepted = {} (total {})'.format(x, eps, n, N_x))
# define std for chains
std = np.std(accepted, axis=0)
logging.info('std for each parameter:{}'.format(std))
for n_job in job_folders:
    np.savetxt(os.path.join(n_job, 'std'), [std])
# Randomly choose starting points for Markov chains
for i in range(N_params):
    unique, counts = np.unique(accepted[:, i], return_counts=True)
    logging.info(f'Number of unique values for {i+1} parameter: {len(unique)} {counts}')
replace = False
if N_chains > n:
    replace = True
    logging.infor("random choice with replacing")
random_ind = np.random.choice(n, N_chains, replace=replace)
C_start = accepted[random_ind]
dist_init = dist[random_ind]
np.set_printoptions(precision=4)
logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
for i, n_job in enumerate(job_folders):
    np.savetxt(os.path.join(n_job, 'C_start'), C_start[i])
    np.savetxt(os.path.join(n_job, 'dist_init'), C_start[i])
    if chain_with_limits:
        np.savetxt(os.path.join(n_job, 'C_limits'), C_limits)

logging.info('2D smooth marginals with {} bins per dimension'.format(num_bin_kde))
if mirror:
    mirrored_data, _ = pp.mirror_data_for_kde(accepted, C_limits[:, 0], C_limits[:, 1])
    print(f"{len(mirrored_data) - len(accepted)} points were added to {len(accepted)} points")
    Z = gaussian_kde_scipy(mirrored_data, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
else:
    # Z = kdepy_fftkde(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
    Z = gaussian_kde_scipy(accepted, C_limits[:, 0], C_limits[:, 1], num_bin_kde)
C_MAP_smooth = find_MAP_kde(Z, C_limits[:, 0], C_limits[:, 1])
print('C_MAP_smooth', C_MAP_smooth)
np.savetxt(os.path.join(job_folders[0], 'C_start'), C_MAP_smooth)
logging.info('Estimated parameters from joint pdf: {}'.format(C_MAP_smooth))
# ##############################################################################
