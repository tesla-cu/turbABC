import numpy as np
from time import time

def covariance_recursive_haario(x, t, cov_prev, mean_prev):
    mean_new = t / (t + 1) * mean_prev + 1 / (t + 1) * x
    cov = (t - 1) / t * cov_prev + \
            1 / t * (t * np.outer(mean_prev, mean_prev) - (t + 1) * np.outer(mean_new, mean_new) + np.outer(x, x))
    return cov, mean_new


def covariance_recursive_paper(x, t, cov_prev, mean_prev):
    delta = (x-mean_prev)
    mean_new = mean_prev + 1 / (t + 1) * delta
    cov = cov_prev + 1/(t+1) * (np.outer(delta, delta) - cov_prev)
    return cov, mean_new


def covariance_recursive_my(x, t, cov_prev, mean_prev):
    delta = (x-mean_prev)
    mean_new = mean_prev + 1 / (t + 1) * delta
    cov = cov_prev + 1/(t+1) * np.outer(delta, delta)- 1/t* cov_prev
    return cov, mean_new

def covariance_recursive_true(x, t):
    mean = np.mean(x)
    cov = 1/t*(np.sum(x**2) - (t+1) * mean**2)
    return cov, mean


x = np.array([100, 125, 200, 303, 50])
mean_prev1 = x[0]
cov_prev1 = 1
mean_prev2 = x[0]
cov_prev2 = 1
mean_prev3 = x[0]
cov_prev3 = 1
for i in range(1, 5):
    a0, b0 = covariance_recursive_true(x[:i+1], i)

    a1, b1 = covariance_recursive_haario(x[i], i, cov_prev1, mean_prev1)
    cov_prev1, mean_prev1 = a1, b1
    a2, b2 = covariance_recursive_paper(x[i], i, cov_prev2, mean_prev2)
    cov_prev2, mean_prev2 = a2, b2
    a3, b3 = covariance_recursive_my(x[i], i, cov_prev3, mean_prev3)
    cov_prev3, mean_prev3 = a3, b3
    print('{} {} {} {}\n{} {} {} {}\n{} {} {}\n{} {} {}\n\n'.format(a0, a1, a2, a3, b0, b1, b2, b3,
                                                                    a1-a0, a2 - a0, a3-a0, b1- b0, b2-b0, b3-b0))


