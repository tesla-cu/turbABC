import numpy as np
import abc_code.utils as utils
import matplotlib.pyplot as plt

data = np.zeros(18)
data[0:6] = np.arange(0.4, 1, 0.1)
data[6:15] = np.arange(1.5, 2.4, 0.1)
data[15:18] = np.arange(2.7, 3, 0.1)
print(data)

Z, max_position = utils.gaussian_kde_scipy(data, [0], [3], 30)

print(Z, max_position)
fig = plt.figure()
ax = plt.gca()
plt.plot(np.linspace(0, 3, 31), Z)
plt.hist(data, bins=30, range=[-0.05, 2.95], normed=1)
fig.savefig('./plots/test_kde')
plt.close('all')

