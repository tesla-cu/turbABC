import numpy as np
import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

folder_scipy = '/home/olga/ABC_MCMC/runs_abc/output_scipykde_20/x_5'
folder_kdepy = '/home/olga/ABC_MCMC/runs_abc/output/x_5.0'
C_limits = np.loadtxt('/home/olga/ABC_MCMC/runs_abc/output/C_limits_init')
limits = np.array([[0.80833332, 1.79166668],
                  [0.62499999, 0.77833334],
                  [1.66999999, 1.99666668],
                  [1.75249999, 2.04750001]])
Z_scipy = np.load(os.path.join(folder_scipy, 'Z.npz'))['Z']
print(Z_scipy.shape)
Z_scipy = Z_scipy/np.sum(Z_scipy)
Z_kdepy = np.load(os.path.join(folder_kdepy, 'Z.npz'))['Z']
print(Z_kdepy.shape)
Z_kdepy = Z_kdepy/np.sum(Z_kdepy)

# diff = Z_scipy-Z_kdepy
fig = plt.figure()
ax = plt.gca()
for i in range(4):
    ind = tuple(np.where(np.arange(4) != i)[0])
    # y = np.sum(diff, axis=ind)
    z = np.sum(Z_scipy, axis=ind)
    z2 = np.sum(Z_kdepy, axis=ind)
    # ax.plot(y, label=str(i))
    x = (np.linspace(C_limits[i, 0], C_limits[i, 1], 21) - limits[i, 0])/(limits[i, 1] - limits[i, 0])
    x2 = (np.linspace(limits[i, 0], limits[i, 1], 21) - limits[i, 0])/(limits[i, 1] - limits[i, 0])
    ax.plot(x, z, label='scypy{}'.format(i))
    ax.plot(x2, z2, label='kdepy{}'.format(i))
plt.legend()
fig.savefig(os.path.join('/home/olga/ABC_MCMC/runs_abc', 'diff'))
plt.close('all')