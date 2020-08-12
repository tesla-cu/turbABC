import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def calc_N(N_total, N_jobs):
    N = np.array([int(N_total / N_jobs)] * N_jobs)
    if N_total % N_jobs != 0:
        reminder = N_total % N_jobs
        N[:reminder] += 1
    return N


path_base = '../overflow_results/'
path = {'output': os.path.join(path_base, 'chains_limits_4/postprocess'), 'visua': os.path.join(path_base, 'plots_limits')}
if not os.path.isdir(path['visua']):
    os.makedirs(path['visua'])

N_jobs = 5
N_samples = 200     # number of samples from posterior
Z = np.load(os.path.join(path['output'], 'Z.npz'))['Z']
limits = np.loadtxt(os.path.join(path['output'], 'C_limits_init'))
Z_max = np.max(Z)
N_params = len(Z.shape)
N_bins = Z.shape[0]
c_max = np.loadtxt(os.path.join(path['output'], f'C_final_smooth{N_bins-1}'))

grid_x = [np.linspace(limits[i, 0], limits[i, 1], N_bins) for i in range(N_params)]
interpolator = RegularGridInterpolator(tuple(grid_x), Z)

ind_array = np.arange(N_bins)
c_array = np.empty((N_samples+1, N_params))
probability = np.empty(N_samples+1)
c_array[0] = c_max
probability[0] = Z_max

print('c_max', c_max)
k = 1
while k < N_samples+1:
    c = np.random.uniform(limits[:, 0], limits[:, 1])
    u = np.random.random()*Z_max
    z = interpolator(c)[0]
    if u <= z:
        c_array[k] = c
        probability[k] = z
        print(probability[k])
        k += 1

np.savetxt(os.path.join(path['output'], 'samples_from_posterior'), c_array)
np.savetxt(os.path.join(path['output'], 'probability_from_posterior'), probability)

N = calc_N(N_samples, N_jobs)
for i in range(N_jobs):
    start, end = np.sum(N[:i]), np.sum(N[:i + 1])
    print('job {}: from {} to {}'.format(i, start, end))
    dir = os.path.join(path['output'], 'calibration_job{}'.format(i))
    if not os.path.isdir(dir):
        os.makedirs(dir)
    np.savetxt(os.path.join(dir, 'c_array_{}'.format(i)), c_array[start:end])

print(f'{N_samples} samples in {N_jobs} jobs')
print(f"{N_samples * 7 / 60 / N_jobs} hours for 1 job")



plt.hist(c_array[:, 0])
plt.show()
plt.hist(c_array[:, 1])
plt.show()
plt.hist(c_array[:, 2])
plt.show()
plt.hist(c_array[:, 3])
plt.show()