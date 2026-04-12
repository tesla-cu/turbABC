import numpy as np
import os
import matplotlib.pyplot as plt

path_base = '../ABC/final/4_params_sigma/'
path = {'output': os.path.join(path_base, 'output'), 'visua': os.path.join(path_base, 'plots')}
if not os.path.isdir(path['visua']):
    os.makedirs(path['visua'])


N = 100
Z = np.load(os.path.join(path['output'], 'Z.npz'))['Z']
ind_max = np.loadtxt(os.path.join(path['output'], 'ind_max'))

Z_max = np.max(Z)
N_params = len(Z.shape)
N_bins = Z.shape[0]

ind_array = np.arange(N_bins)
x = np.zeros(N_params, dtype=np.int32)
samples_ind = np.empty((N+1, N_params), dtype=np.int32)
probability = np.empty(N+1)
samples_ind[0] = ind_max
probability[0] = Z_max

k = 0
while k < N:
    for i in range(N_params):
        x[i] = np.random.choice(ind_array, replace=True)
    u = np.random.random()*Z_max
    z = Z[tuple(x)]
    print(z, Z_max)
    if u <= z:
        samples_ind[k+1] = x
        probability[k+1] = z
        print(probability[k+1])
        k += 1


accepted = np.load(os.path.join(path['output'], 'accepted_0.npz'))['C']
C_limits = np.zeros((N_params, 2))
if np.max(accepted[:, 0]) < 0.0:
    C_limits[0] = [np.min(accepted[:, 0]), np.max(accepted[:, 0])]
else:
    C_limits[0] = [np.min(accepted[:, 0]), 0.0]
for i in range(1, N_params):
    C_limits[i] = [np.min(accepted[:, i]), np.max(accepted[:, i])]

C = np.empty((N_params, N_bins))
for i in range(N_params):
    C[i] = np.linspace(C_limits[i, 0], C_limits[i, 1], N_bins)
C = C.T
samples = np.empty((N+1, N_params))

for i in range(N+1):
    for j in range(N_params):
        samples[i, j] = C[samples_ind[i, j], j]


np.savetxt(os.path.join(path['output'], 'samples_from_posterior'), samples)
np.savetxt(os.path.join(path['output'], 'probability_from_posterior'), probability)

plt.hist(samples[:, 0], 20)
plt.show()
plt.hist(samples[:, 1], 20)
plt.show()
plt.hist(samples[:, 2], 20)
plt.show()