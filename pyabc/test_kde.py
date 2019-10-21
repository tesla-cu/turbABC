import unittest
import numpy as np
import pyabc.kde as kde
from scipy.stats.distributions import norm
# import matplotlib.pyplot as plt
np.random.seed(0)


class KDETest(unittest.TestCase):
    def test_grid(self):
        # 1d array
        c = np.linspace(-1, 1, 6, endpoint=True)
        c_mesh, c_ravel = kde.grid_for_kde(-1, 1, 5)
        np.testing.assert_array_almost_equal(c, c_mesh[0])
        np.testing.assert_array_almost_equal(c, c_ravel[0])
        # 2d
        c_mesh_true = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
        c_ravel_true = [[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3]]
        c_mesh, c_ravel = kde.grid_for_kde([1, 1], [3, 3], 2)
        np.testing.assert_array_almost_equal(c_mesh_true, c_mesh)
        np.testing.assert_array_almost_equal(c_ravel_true, c_ravel)

    def test_MAP(self):
        # 1d array, single MAP
        c = np.linspace(-1, 1, 6, endpoint=True)
        Z = np.array([1.1, 2.3, 3.5, 1.3, 9.2, 2.1])
        c_map = kde.find_MAP_kde(Z, -1, 1)
        np.testing.assert_array_almost_equal(c_map, [[c[4]]])

        # 1d array, multiple MAP
        Z = np.array([1.1, 2.3, 3.5, 1.3, 3.5, 2.1])
        c_map = kde.find_MAP_kde(Z, -1, 1)
        np.testing.assert_array_almost_equal(c_map, [[c[2]], [c[4]]])

        # 2d array, single MAP
        c1 = np.linspace(-1, 1, 6, endpoint=True)
        c2 = np.linspace(-3, 4, 6, endpoint=True)
        Z = np.array([[1.1, 2.3, 3.5, 1.3, 9.2, 2.1], [1.1, 2.3, 3.5, 1.3, 9.2, 2.1], [1.1, 2.3, 3.5, 1.3, 9.2, 2.1],
             [1.1, 2.3, 3.5, 1.3, 9.2, 2.1], [1.1, 9.8, 3.5, 1.3, 9.2, 2.1], [1.1, 2.3, 3.5, 1.3, 9.2, 2.1]])
        c_map = kde.find_MAP_kde(Z, [-1, -3], [1, 4])
        np.testing.assert_array_almost_equal(c_map, [[c1[4], c2[1]]])

        # 2d array, multiple MAP
        Z = np.array([[1.1, 2.3, 9.8, 1.3, 9.2, 2.1], [1.1, 2.3, 3.5, 1.3, 9.2, 2.1], [1.1, 2.3, 3.5, 1.3, 9.2, 2.1],
             [1.1, 2.3, 3.5, 1.3, 9.2, 2.1], [1.1, 9.8, 3.5, 1.3, 9.2, 2.1], [1.1, 2.3, 3.5, 1.3, 9.2, 2.1]])
        c_map = kde.find_MAP_kde(Z, [-1, -3], [1, 4])
        np.testing.assert_array_almost_equal(c_map, [[c1[0], c2[2]], [c1[4], c2[1]]])

    def test_kde1d(self):

        num_bin = 100
        data = np.concatenate([norm(-1, 1.).rvs(1600), norm(1, 0.3).rvs(400)]).reshape((-1, 1))
        a, b = [-4.5], [3.5]
        x_grid = np.linspace(a[0], b[0], num_bin + 1)
        pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))

        Z_scipy = kde.gaussian_kde_scipy(data, a, b, num_bin)
        Z_kdepy = kde.kdepy_fftkde(data, a, b, num_bin)
        # Normalize to summ up to 1
        Z_scipy = Z_scipy / np.trapz(Z_scipy, x=x_grid)
        Z_kdepy = Z_kdepy / np.trapz(Z_kdepy, x=x_grid)
        # find MAP values
        MAP_scipy = kde.find_MAP_kde(Z_scipy, a, b)[0]
        MAP_kdepy = kde.find_MAP_kde(Z_kdepy, a, b)[0]

        np.testing.assert_array_almost_equal(Z_scipy, Z_kdepy, decimal=3)
        np.testing.assert_array_almost_equal(MAP_scipy, x_grid[np.argmax(pdf_true)])
        np.testing.assert_array_almost_equal(MAP_kdepy, x_grid[np.argmax(pdf_true)])
        np.testing.assert_array_almost_equal(MAP_scipy, MAP_kdepy)

        # print(np.sum(Z_scipy), np.sum(Z_kdepy))
        # print(np.mean(np.abs(Z_scipy - Z_kdepy)))
        # plt.hist(data[:, 0], bins=100, alpha=0.3, density=1)
        # plt.plot(x_grid, Z_scipy, label='scipy')
        # plt.plot(x_grid, Z_kdepy, label='fft')
        # plt.legend()
        # plt.show()
        # plt.plot(Z_scipy-Z_kdepy)
        # plt.show()

    def test_kde2d(self):
        num_bin_joint = 100
        data = np.random.multivariate_normal((3, 3), [[0.8, 0.05], [0.05, 0.7]], 100)
        a = [0, 0]
        b = [6, 6]
        grid, _ = kde.grid_for_kde(a, b, num_bin_joint)
        Z_scipy = kde.gaussian_kde_scipy(data, a, b, num_bin_joint)
        Z_kdepy = kde.kdepy_fftkde(data, a, b, num_bin_joint)
        # Normalize to summ up to 1
        Z_scipy = Z_scipy / np.trapz(np.trapz(Z_scipy, x=grid[0], axis=0), grid[1][0, :])
        Z_kdepy = Z_kdepy / np.trapz(np.trapz(Z_kdepy, x=grid[0], axis=0), grid[1][0, :])
        # find MAP values
        MAP_scipy = kde.find_MAP_kde(Z_scipy, a, b)
        MAP_kdepy = kde.find_MAP_kde(Z_kdepy, a, b)

        np.testing.assert_array_almost_equal(Z_scipy, Z_kdepy, decimal=2)
        np.testing.assert_array_almost_equal(MAP_scipy, MAP_kdepy, decimal=1)

        # print(np.sum(Z_scipy), np.sum(Z_kdepy))
        # print(np.mean(np.abs(Z_scipy - Z_kdepy)))
        # print(MAP_kdepy, MAP_scipy)


if __name__ == '__main__':
    unittest.main()
